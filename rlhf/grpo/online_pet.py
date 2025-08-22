import os
from collections import deque
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from accelerate import DeepSpeedPlugin
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.optimization import Adafactor
import random
import wandb
import gc

from reward_models.load_datasets import load_train_eval_dataset
from reward_utils import get_reward
from accelerate.utils import DummyOptim, DummyScheduler


@dataclass
class OnlinePETConfig:
    """
    Configuration for the online pessimistic reward model update.
    """
    online_pet_enabled: bool = field(default=False, metadata={"help": "Enable online PET updates."})
    k_top_responses: int = field(default=1, metadata={
        "help": "Number of top responses to store from each batch for adversarial training."})
    update_interval_steps: int = field(default=16,
                                       metadata={"help": "Number of GRPO steps between each reward model update."})
    eval_online_pet_every: int = field(default=8, metadata={"help": "How many PET steps between evaluations."})
    rm_update_steps: int = field(default=1,
                                 metadata={"help": "Number of epochs to train RM on collected data."})
    pessimistic_gradient_accumulation_steps: int = field(default=32, metadata={"help": "Gradient accumulation steps for pessimistic loss."})
    bt_gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps for BT loss."})
    rm_gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing for reward models."})
    rm_update_learning_rate: float = field(default=4e-5, metadata={"help": "Learning rate for the reward model optimizer."})
    pessimistic_loss_weight: float = field(default=0.1, metadata={"help": "Weight for the pessimistic loss component."})
    bt_loss_weight: float = field(default=1.0, metadata={"help": "Weight for the BT loss component."})
    preference_dataset_path: str = field(default="", metadata={"help": "Path to the original preference dataset for BT loss."})
    preference_batch_size: int = field(default=1, metadata={"help": "Batch size for the preference dataset dataloader."})
    adversarial_batch_size: int = field(default=1, metadata={"help": "Batch size for the adversarial examples."})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for the evaluation dataloader."})
    rm_save_path: str = field(default="", metadata={"help": "Path to save the reward model checkpoints. If empty, no saving."})
    rm_optimizer: str = field(default="AdamW", metadata={"help": "Optimizer to use for RM update. Values: AdamW, Adafactor"})
    rm_buffer_size: int = field(default=32, metadata={"help": "Buffer size for RM updates."})
    rm_deepspeed_plugin: str = field(default="", metadata={"help": "Deepspeed plugin config path for RM training"})


class OnlinePETCallback(TrainerCallback):
    def __init__(self, pet_config, accelerator, reward_models, reward_tokenizers, reward_controller, policy_tokenizer, model_name):
        self.pet_config = pet_config
        self.accelerator = accelerator
        self.reward_models = reward_models
        self.reward_tokenizers = reward_tokenizers
        self.reward_controller = reward_controller
        self.policy_tokenizer = policy_tokenizer
        self.model_name = model_name
        self.pet_update_counter = 0
        self.adversarial_buffer = deque(maxlen=pet_config.rm_buffer_size)
        self.num_preference_epochs = 0
        self.initialized = False

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
        self._initialize_rm_training()
        self.initialized = True

    def _initialize_rm_training(self):
        if not self.pet_config.online_pet_enabled or self.accelerator is None:
            return
        
        print("--- Initializing Reward Model Training Components ---")
        if self.pet_config.rm_gradient_checkpointing:
            for rm in self.reward_models:
                rm.gradient_checkpointing_enable(use_reentrant=False)


        policy_plugin = self.accelerator.state.deepspeed_plugin
        rm_plugin = DeepSpeedPlugin(hf_ds_config=self.pet_config.rm_deepspeed_plugin)
        deepspeed_plugins = {
            "policy": policy_plugin,
            "rm": rm_plugin,
        }
        self.accelerator.state.deepspeed_plugins = deepspeed_plugins
        self.accelerator.state.select_deepspeed_plugin("policy")

        rm_tokenizer = self.reward_tokenizers[0]
        rm_tokenizer.max_length = 1024
        preference_dataset, eval_dataset = load_train_eval_dataset(
            data_path=self.pet_config.preference_dataset_path,
            tokenizer=rm_tokenizer,
            model_name=self.model_name
        )
        self.preference_dataloader = DataLoader(
            preference_dataset,
            batch_size=self.pet_config.preference_batch_size,
            shuffle=True,
            collate_fn=self.collate_preference_data
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.pet_config.eval_batch_size,
            collate_fn=self.collate_preference_data
        )

        params = [p for rm in self.reward_models for p in rm.parameters() if p.requires_grad]
        self.rm_optimizer = Adafactor(
            params,
            lr=self.pet_config.rm_update_learning_rate,
            decay_rate=-0.8,
            weight_decay=0.0,
            scale_parameter=False,
            relative_step=False,
        ) if str.lower(self.pet_config.rm_optimizer) == 'adafactor' else torch.optim.AdamW(
            params,
            lr=self.pet_config.rm_update_learning_rate,
        ) if str.lower(self.pet_config.rm_optimizer) == 'adamw' else None
        assert self.rm_optimizer is not None, f"Unsupported optimizer {self.pet_config.rm_optimizer}"
        self.reward_models[0], self.rm_optimizer, self.preference_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.reward_models[0], self.rm_optimizer, self.preference_dataloader, self.eval_dataloader
        )
        self.preference_data_iterator = iter(self.preference_dataloader)


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.pet_config.online_pet_enabled or not self.pet_config.rm_save_path or not self.initialized:
            return

        if not self.accelerator.is_main_process:
            return
        current_epoch = int(state.epoch)
        print(f"--- Saving reward model at epoch {current_epoch} (step {state.global_step}) ---")

        for i, (rm, r_tok) in enumerate(zip(self.reward_models, self.reward_tokenizers)):
            epoch_save_path = os.path.join(self.pet_config.rm_save_path, f"epoch_{current_epoch}")
            rm_save_path = os.path.join(epoch_save_path, f"reward_model_{i}")

            unwrapped_rm = self.accelerator.unwrap_model(rm)

            unwrapped_rm.save_pretrained(rm_save_path)
            r_tok.save_pretrained(rm_save_path)
            print(f"Reward model {i} saved to {rm_save_path}")

    def collate_preference_data(self, batch):
        rm_tokenizer = self.reward_tokenizers[0]
        return {
            'input_ids_chosen': torch.nn.utils.rnn.pad_sequence([item['input_ids_chosen'] for item in batch], batch_first=True, padding_value=rm_tokenizer.pad_token_id, padding_side='left'),
            'attention_mask_chosen': torch.nn.utils.rnn.pad_sequence([item['attention_mask_chosen'] for item in batch], batch_first=True, padding_value=0, padding_side='left'),
            'input_ids_rejected': torch.nn.utils.rnn.pad_sequence([item['input_ids_rejected'] for item in batch], batch_first=True, padding_value=rm_tokenizer.pad_token_id, padding_side='left'),
            'attention_mask_rejected': torch.nn.utils.rnn.pad_sequence([item['attention_mask_rejected'] for item in batch], batch_first=True, padding_value=0, padding_side='left'),
        }

    def _get_preference_batch(self):
        try:
            return next(self.preference_data_iterator)
        except StopIteration:
            self.preference_data_iterator = iter(self.preference_dataloader)
            self.num_preference_epochs += 1
            return next(self.preference_data_iterator)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.pet_config.online_pet_enabled or not self.accelerator.is_main_process or not self.initialized:
            return

        is_update_step = (state.global_step > 0 and state.global_step % self.pet_config.update_interval_steps == 0)

        if is_update_step:
            print(f"--- Step {state.global_step}: Starting Online PET Actions ---")
            self._perform_rm_update(state.global_step)
            self.pet_update_counter += 1

            if self.pet_update_counter % self.pet_config.eval_online_pet_every == 0:
                self._evaluate_rm(state.global_step)

            print(f"--- Finished Online PET Actions ---")
            torch.cuda.empty_cache()

    def _evaluate_rm(self, step):
        print("--- Evaluating Reward Model ---")
        rm = self.reward_models[0]
        rm.eval()
        total_correct = 0
        total_samples = 0
        total_bt_loss = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                chosen_rewards = rm(
                    input_ids=batch['input_ids_chosen'],
                    attention_mask=batch['attention_mask_chosen']
                ).logits.squeeze(-1)

                rejected_rewards = rm(
                    input_ids=batch['input_ids_rejected'],
                    attention_mask=batch['attention_mask_rejected']
                ).logits.squeeze(-1)

                total_bt_loss += -F.logsigmoid(chosen_rewards - rejected_rewards).mean().item()
                total_correct += (chosen_rewards > rejected_rewards).sum().item()
                total_samples += chosen_rewards.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        bt_loss = total_bt_loss / len(self.eval_dataloader) if len(self.eval_dataloader) > 0 else 0
        print(f"RM Evaluation Accuracy: {accuracy:.4f}")
        if wandb.run:
            wandb.log({
                "eval/rm_accuracy": accuracy,
                "eval/rm_bt_loss": bt_loss,
            }, step=wandb.run.step)
        rm.train()
        print("--- Finished RM Evaluation ---")

    def _perform_rm_update(self, step):
        print("--- Starting RM Update ---")
        new_adversarial_data = self.reward_controller.get_and_clear_adversarial_buffer()
        self.adversarial_buffer.extend(new_adversarial_data)

        effective_adv_batch_size = self.pet_config.adversarial_batch_size * self.pet_config.pessimistic_gradient_accumulation_steps
        if len(self.adversarial_buffer) < effective_adv_batch_size:
            print(
                f"Not enough adversarial samples to form a full batch. Have {len(self.adversarial_buffer)}, need {effective_adv_batch_size}. Storing for next update.")
            return

        self.accelerator.state.select_deepspeed_plugin("rm")

        num_adversarial_samples = len(self.adversarial_buffer)
        if wandb.run:
            wandb.log({"update/adversarial_samples": num_adversarial_samples}, step=wandb.run.step)

        rm = self.reward_models[0]
        rm_tokenizer = self.reward_tokenizers[0]
        rm.train()

        for epoch in range(self.pet_config.rm_update_steps):
            adv_buffer_list = list(self.adversarial_buffer)
            random.shuffle(adv_buffer_list)
            print(f"RM Update Epoch {epoch + 1}/{self.pet_config.rm_update_steps}")

            num_adv_micro_batches = len(adv_buffer_list) // self.pet_config.adversarial_batch_size
            num_optimizer_steps = num_adv_micro_batches // self.pet_config.pessimistic_gradient_accumulation_steps

            if num_optimizer_steps == 0:
                print("Not enough batches for one optimizer step, skipping epoch.")
                continue

            adv_buffer_iterator = iter(adv_buffer_list)
            def next_adv_batch():
                nonlocal adv_buffer_iterator
                try:
                    return [next(adv_buffer_iterator) for _ in range(self.pet_config.adversarial_batch_size)]
                except StopIteration:
                    adv_buffer_iterator = iter(adv_buffer_list)
                    return [next(adv_buffer_iterator) for _ in range(self.pet_config.adversarial_batch_size)]

            for opt_step in range(num_optimizer_steps):
                # self.rm_optimizer.zero_grad()

                # --- Pessimistic Loss ---
                pessimistic_loss_item = 0
                i = 0
                while i < self.pet_config.pessimistic_gradient_accumulation_steps:
                    adv_batch = next_adv_batch()

                    adv_prompts, adv_responses, _, adv_ref_rewards = zip(*adv_batch)
                    texts = [p + c for p, c in zip(adv_prompts, adv_responses)]
                    adv_rewards_new = get_reward(rm, rm_tokenizer, adv_prompts, adv_responses, texts, require_grad=True)

                    pessimistic_loss = torch.tensor(0.0, device=self.accelerator.device)
                    if adv_ref_rewards[0] is not None:
                        adv_ref_rewards = torch.tensor(adv_ref_rewards, device=self.accelerator.device)
                        mask = adv_rewards_new > adv_ref_rewards
                        if mask.any():
                            filtered_adv_rewards = adv_rewards_new[mask]
                            filtered_ref_rewards = adv_ref_rewards[mask]
                            pessimistic_loss = (filtered_adv_rewards - filtered_ref_rewards).mean()
                    else:
                        pessimistic_loss = adv_rewards_new.mean()

                    pessimistic_loss *= self.pet_config.pessimistic_loss_weight
                    pessimistic_loss_item += pessimistic_loss.item()

                    scaled_pessimistic_loss = pessimistic_loss * 2 # / self.pet_config.pessimistic_gradient_accumulation_steps
                    # deepspeed scales the loss by the gradient accumulation steps
                    if scaled_pessimistic_loss.requires_grad:
                        self.accelerator.backward(scaled_pessimistic_loss)
                        print("Info: scaled_pessimistic_loss does require gradients")
                        i += 1
                    else:
                        print("Warning: scaled_pessimistic_loss does not require gradients, skipping backward pass.")

                del pessimistic_loss, scaled_pessimistic_loss, adv_rewards_new, adv_ref_rewards

                # --- BT Loss on Preference Data ---
                bt_loss_item = 0
                bt_accuracy = 0
                if self.pet_config.bt_gradient_accumulation_steps > 0:
                    i = 0
                    while i < self.pet_config.bt_gradient_accumulation_steps:
                        preference_batch = self._get_preference_batch()

                        chosen_rewards = rm(
                            input_ids=preference_batch['input_ids_chosen'].to(self.accelerator.device),
                            attention_mask=preference_batch['attention_mask_chosen'].to(self.accelerator.device)
                        ).logits.squeeze(-1)

                        rejected_rewards = rm(
                            input_ids=preference_batch['input_ids_rejected'].to(self.accelerator.device),
                            attention_mask=preference_batch['attention_mask_rejected'].to(
                                self.accelerator.device)
                        ).logits.squeeze(-1)

                        bt_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                        bt_loss *= self.pet_config.bt_loss_weight

                        bt_accuracy += (chosen_rewards > rejected_rewards).float().mean().item()
                        bt_loss_item += bt_loss.item()

                        scaled_bt_loss = bt_loss * 2 # / self.pet_config.bt_gradient_accumulation_steps
                        # deepspeed scales the loss by the gradient accumulation steps
                        if scaled_bt_loss.requires_grad:
                            self.accelerator.backward(scaled_bt_loss)
                            i += 1
                        else:
                            print("Warning: scaled_bt_loss does not require gradients, skipping backward pass.")

                    del bt_loss, scaled_bt_loss, chosen_rewards, rejected_rewards

                # # Optimizer step and logging
                # self.rm_optimizer.step()

                avg_pess_loss = pessimistic_loss_item / self.pet_config.pessimistic_gradient_accumulation_steps
                avg_bt_loss = (bt_loss_item / self.pet_config.bt_gradient_accumulation_steps) if self.pet_config.bt_gradient_accumulation_steps > 0 else 0
                avg_bt_accuracy = (bt_accuracy / self.pet_config.bt_gradient_accumulation_steps) if self.pet_config.bt_gradient_accumulation_steps > 0 else 0
                total_avg_loss = avg_pess_loss + avg_bt_loss

                print(
                    f"  Step {opt_step + 1}/{num_optimizer_steps}: Pessimistic Loss: {avg_pess_loss:.4f}, BT Loss: {avg_bt_loss:.4f}, Total Loss: {total_avg_loss:.4f}, BT Accuracy: {avg_bt_accuracy:.4f}")

                if wandb.run:
                    log_data = {
                        "update/pessimistic_loss": avg_pess_loss,
                        "update/bt_loss": avg_bt_loss,
                        "update/total_loss": total_avg_loss,
                        "update/bt_accuracy": avg_bt_accuracy,
                        "update/num_preference_epochs": self.num_preference_epochs,
                    }
                    wandb.log(log_data, step=wandb.run.step)

        print("--- RM Update Finished ---")
        self.accelerator.state.select_deepspeed_plugin("policy")
        rm.eval()