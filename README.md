# Generalizable Reward Model for LLMs
Code for NeurIPS 2024 paper ["Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs"](https://arxiv.org/abs/2406.10216). Our open-sourced reward models are available at [🤗 huggingface](https://huggingface.co/collections/Ray2333/grm-66882bdf7152951779506c7b).

This repo is under preparation. The evaluation code and BoN/PPO code will be added soon.


## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```

Then, go the `scripts' folder and train the reward model with the default hyperparameters
```
cd scripts
sh train_bt_rm_full.sh
sh train_bt_rm_lora.sh
sh train_grm_full.sh
sh train_grm_lora.sh
```







## Citation

```
@inproceedings{yang2024regularizing,
  title={Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs},
  author={Yang, Rui and Ding, Ruomeng and Lin, Yong and Zhang, Huan and Zhang, Tong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
