# Generalizable-Reward-Model
Code for NeurIPS 2024 paper "Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs". Our open-sourced reward models are available at [https://huggingface.co/collections/Ray2333/grm](https://huggingface.co/collections/Ray2333/grm-66882bdf7152951779506c7b).

This repo is under preparation. The evaluation code and BoN/PPO code will be added soon.


## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```

Then, train the reward model with the default hyperparameters
```
cd reward_models
CUDA_VISIBLE_DEVICES=0,1 accelerate launch run_reward_models_train.py 
CUDA_VISIBLE_DEVICES=0,1 accelerate launch run_grm_reward_train.py 
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
