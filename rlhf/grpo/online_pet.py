import os
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.optimization import Adafactor
import random
import wandb
import gc

from reward_models.load_datasets import load_train_eval_dataset
from reward_utils import get_reward


@dataclass
class OnlinePETConfig:
    """
    Configuration for the online pessimistic reward model update.
    """
    online_pet_enabled: bool = field(default=False, metadata={"help": "Enable online PET updates."})
    move_policy_to_cpu: bool = field(default=False, metadata={"help": "Move policy to CPU during RM update to save memory."})
    move_rm_to_cpu: bool = field(default=False, metadata={"help": "Move reward model to CPU when not in use."})
    k_top_responses: int = field(default=1, metadata={
        "help": "Number of top responses to store from each batch for adversarial training."})
    update_interval_steps: int = field(default=32,
                                       metadata={"help": "Number of GRPO steps between each reward model update."})
    eval_online_pet_every: int = field(default=4, metadata={"help": "How many PET steps between evaluations."})
    rm_update_steps: int = field(default=1,
                                 metadata={"help": "Number of epochs to train RM on collected data."})
    rm_gradient_accumulation_steps: int = field(default=32, metadata={"help": "Gradient accumulation steps for RM update."})
    rm_gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing for reward models."})
    rm_update_learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate for the reward model optimizer."})
    pessimistic_loss_weight: float = field(default=0.1, metadata={"help": "Weight for the pessimistic loss component."})
    bt_loss_weight: float = field(default=1.0, metadata={"help": "Weight for the BT loss component."})
    preference_dataset_path: str = field(default="", metadata={"help": "Path to the original preference dataset for BT loss."})
    preference_batch_size: int = field(default=1, metadata={"help": "Batch size for the preference dataset dataloader."})
    adversarial_batch_size: int = field(default=1, metadata={"help": "Batch size for the adversarial examples."})
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size for the evaluation dataloader."})
    rm_save_path: str = field(default="", metadata={"help": "Path to save the reward model checkpoints. If empty, no saving."})


class OnlinePETCallback(TrainerCallback):
    def __init__(self, pet_config, accelerator, reward_models, reward_tokenizers, reward_controller, policy_tokenizer, model_name):
        self.pet_config = pet_config
        self.accelerator = accelerator
        self.reward_models = reward_models
        self.reward_tokenizers = reward_tokenizers
        self.reward_controller = reward_controller
        self.policy_tokenizer = policy_tokenizer
        self.model_name = model_name
        self.adversarial_leftovers = []
        self.pet_update_counter = 0

        if pet_config.online_pet_enabled:
            if self.pet_config.rm_gradient_checkpointing:
                for rm in self.reward_models:
                    rm.gradient_checkpointing_enable()

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
            self.preference_data_iterator = iter(self.preference_dataloader)

            self.rm_optimizer = Adafactor(
                [p for rm in self.reward_models for p in rm.parameters() if p.requires_grad],
                lr=self.pet_config.rm_update_learning_rate,
                decay_rate=-0.8,
                weight_decay=0.0,
                scale_parameter=False,
                relative_step=False,
            )
            if self.pet_config.move_rm_to_cpu:
                for rm in self.reward_models:
                    rm.to("cpu")
                self._move_optimizer_to_device(self.rm_optimizer, torch.device("cpu"))

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.pet_config.online_pet_enabled or not self.pet_config.rm_save_path:
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
            return next(self.preference_data_iterator)

    def _move_optimizer_to_device(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.pet_config.online_pet_enabled or not self.accelerator.is_main_process:
            return

        if self.pet_config.move_rm_to_cpu:
            for rm in self.reward_models:
                rm.to(self.accelerator.device)
            self._move_optimizer_to_device(self.rm_optimizer, self.accelerator.device)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.pet_config.online_pet_enabled or not self.accelerator.is_main_process:
            return

        is_update_step = (state.global_step > 0 and state.global_step % self.pet_config.update_interval_steps == 0)

        if is_update_step:
            print(f"\n--- Step {state.global_step}: Starting Online PET Actions ---")
            policy = self.accelerator.unwrap_model(kwargs['model'])
            policy_optimizer = kwargs['optimizer']

            if self.pet_config.move_policy_to_cpu:
                self._move_optimizer_to_device(policy_optimizer, torch.device("cpu"))
                policy.to('cpu')
            torch.cuda.empty_cache()

            self._perform_rm_update(state.global_step)
            self.pet_update_counter += 1

            if self.pet_update_counter % self.pet_config.eval_online_pet_every == 0:
                self._evaluate_rm(state.global_step)

            if self.pet_config.move_policy_to_cpu:
                policy.to(self.accelerator.device)
                self._move_optimizer_to_device(policy_optimizer, self.accelerator.device)

            print(f"--- Finished Online PET Actions ---")

            if self.pet_config.move_rm_to_cpu:
                for rm in self.reward_models:
                    rm.to('cpu')
                self._move_optimizer_to_device(self.rm_optimizer, torch.device("cpu"))
            torch.cuda.empty_cache()

    def _evaluate_rm(self, step):
        print("--- Evaluating Reward Model ---")
        rm = self.reward_models[0]
        rm.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                chosen_rewards = rm(
                    input_ids=batch['input_ids_chosen'].to(self.accelerator.device),
                    attention_mask=batch['attention_mask_chosen'].to(self.accelerator.device)
                ).logits.squeeze(-1)

                rejected_rewards = rm(
                    input_ids=batch['input_ids_rejected'].to(self.accelerator.device),
                    attention_mask=batch['attention_mask_rejected'].to(self.accelerator.device)
                ).logits.squeeze(-1)
                
                total_correct += (chosen_rewards > rejected_rewards).sum().item()
                total_samples += chosen_rewards.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"RM Evaluation Accuracy: {accuracy:.4f}")
        if wandb.run:
            wandb.log({"eval/rm_accuracy": accuracy}, step=wandb.run.step)
        rm.train()
        print("--- Finished RM Evaluation ---")

    def _perform_rm_update(self, step):
        print("--- Starting RM Update ---")
        new_adversarial_data = self.reward_controller.get_and_clear_adversarial_buffer()
        all_adversarial_data = self.adversarial_leftovers + new_adversarial_data

        effective_adv_batch_size = self.pet_config.adversarial_batch_size * self.pet_config.rm_gradient_accumulation_steps
        if len(all_adversarial_data) < effective_adv_batch_size:
            print(f"Not enough adversarial samples to form a full batch. Have {len(all_adversarial_data)}, need {effective_adv_batch_size}. Storing for next update.")
            self.adversarial_leftovers = all_adversarial_data
            return

        num_adversarial_samples = len(all_adversarial_data)
        if wandb.run:
            wandb.log({"update/adversarial_samples": num_adversarial_samples}, step=wandb.run.step)

        for epoch in range(self.pet_config.rm_update_steps):
            random.shuffle(all_adversarial_data)
            print(f"RM Update Epoch {epoch + 1}/{self.pet_config.rm_update_steps}")

            num_batches = len(all_adversarial_data) // self.pet_config.adversarial_batch_size
            num_optimizer_steps = num_batches // self.pet_config.rm_gradient_accumulation_steps

            if num_optimizer_steps == 0:
                print("Not enough batches for one optimizer step, skipping epoch.")
                continue

            self.rm_optimizer.zero_grad()
            bt_accuracy = 0
            total_loss_item = 0
            pessimistic_loss_item = 0
            bt_loss_item = 0
            for i in range(num_batches):
                start_idx = i * self.pet_config.adversarial_batch_size
                end_idx = start_idx + self.pet_config.adversarial_batch_size
                adv_batch = all_adversarial_data[start_idx:end_idx]


                rm = self.reward_models[0]

                # --- Pessimistic Loss ---
                adv_prompts, adv_responses, _, adv_ref_rewards = zip(*adv_batch)
                rm_tokenizer = self.reward_tokenizers[0]
                texts = [p + c for p, c in zip(adv_prompts, adv_responses)]
                adv_rewards_new = get_reward(rm, rm_tokenizer, adv_prompts, adv_responses, texts)

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

                # Scale and backward pessimistic loss
                scaled_pessimistic_loss = pessimistic_loss
                if self.pet_config.rm_gradient_accumulation_steps > 1:
                    scaled_pessimistic_loss = scaled_pessimistic_loss / self.pet_config.rm_gradient_accumulation_steps

                if scaled_pessimistic_loss.requires_grad:
                    self.accelerator.backward(scaled_pessimistic_loss, retain_graph=False)

                # Free memory from pessimistic loss calculation
                del pessimistic_loss, scaled_pessimistic_loss, adv_rewards_new, adv_ref_rewards
                torch.cuda.empty_cache()

                preference_batch = self._get_preference_batch()

                # --- BT Loss on Preference Data ---
                chosen_rewards = rm(
                    input_ids=preference_batch['input_ids_chosen'].to(self.accelerator.device),
                    attention_mask=preference_batch['attention_mask_chosen'].to(self.accelerator.device)
                ).logits.squeeze(-1)

                rejected_rewards = rm(
                    input_ids=preference_batch['input_ids_rejected'].to(self.accelerator.device),
                    attention_mask=preference_batch['attention_mask_rejected'].to(self.accelerator.device)
                ).logits.squeeze(-1)

                bt_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                bt_loss *= self.pet_config.bt_loss_weight

                bt_accuracy += (chosen_rewards > rejected_rewards).float().mean().item()
                bt_loss_item += bt_loss.item()

                # Scale and backward BT loss
                scaled_bt_loss = bt_loss
                if self.pet_config.rm_gradient_accumulation_steps > 1:
                    scaled_bt_loss = scaled_bt_loss / self.pet_config.rm_gradient_accumulation_steps

                if scaled_bt_loss.requires_grad:
                    self.accelerator.backward(scaled_bt_loss, retain_graph=False)
                
                del bt_loss, scaled_bt_loss, chosen_rewards, rejected_rewards

                # Log metrics
                total_loss_item += pessimistic_loss_item + bt_loss_item


                if (i + 1) % self.pet_config.rm_gradient_accumulation_steps == 0:
                    print(f"  Step {(i + 1) // self.pet_config.rm_gradient_accumulation_steps}/{num_optimizer_steps}: Pessimistic Loss: {pessimistic_loss_item:.4f}, BT Loss: {bt_loss_item:.4f}, Total Loss: {total_loss_item:.4f}, BT Accuracy: {bt_accuracy:.4f}")
                    self.rm_optimizer.step()
                    self.rm_optimizer.zero_grad()
                    if wandb.run:
                        log_data = {
                            "update/pessimistic_loss": pessimistic_loss_item / self.pet_config.rm_gradient_accumulation_steps,
                            "update/bt_loss": bt_loss_item / self.pet_config.rm_gradient_accumulation_steps,
                            "update/total_loss": total_loss_item / self.pet_config.rm_gradient_accumulation_steps,
                            "update/bt_accuracy": bt_accuracy / self.pet_config.rm_gradient_accumulation_steps,
                        }
                        wandb.log(log_data, step=wandb.run.step)

            processed_batches = num_optimizer_steps * self.pet_config.rm_gradient_accumulation_steps
            self.adversarial_leftovers = all_adversarial_data[processed_batches * self.pet_config.adversarial_batch_size:]
            print(f"Stored {len(self.adversarial_leftovers)} adversarial samples for next update.")

        print("--- RM Update Finished ---")