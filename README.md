# Generalizable Reward Model for LLMs
Code for NeurIPS 2024 paper ["Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs"](https://arxiv.org/abs/2406.10216). Our open-sourced reward models are available at [🤗 huggingface](https://huggingface.co/collections/Ray2333/grm-66882bdf7152951779506c7b).

This repo is under preparation. The PPO code will be added soon.

## Models

Check out our GRM series below, which are evlauated on [reward-bench](https://huggingface.co/spaces/allenai/reward-bench).



|       Model               | Average       |  Chat     |     Chat Hard      |     Safety      |     Reasoning     |   
|:-------------------------:|:-------------:|:---------:|:---------:|:--------:|:-----------:|
|[GRM-Llama3-8B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Llama3-8B-rewardmodel-ft)**(8B)**|91.5|95.5|86.2|90.8|93.6|
|[GRM-Llama3.2-3B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft)**(3B)**|90.9|91.6|84.9|92.7|94.6|
| [GRM-gemma2-2B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft) **(2B)**| 88.4 | 93.0 | 77.2 | 92.2 | 91.2 |
|[GRM-llama3-8B-sftreg](https://huggingface.co/Ray2333/GRM-llama3-8B-sftreg)**(8B)**|87.0|98.6|67.8|89.2|92.3|
|[GRM-llama3.2-3B-sftreg](https://huggingface.co/Ray2333/GRM-llama3.2-3B-sftreg)**(3B)**|85.8|96.4|67.1|88.2|91.6|
|[GRM-Gemma-2B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Gemma-2B-rewardmodel-ft) **(2B)**|  84.7 | 89.4 | 75.2 | 85.5 | 88.8 |
|  [GRM-Gemma2-2B-sftreg](https://huggingface.co/Ray2333/GRM-Gemma2-2B-sftreg)**(2B)** | 81.0 |  97.2    |  59.6 | 86.9 |   80.3 |
|  [GRM-Gemma-2B-sftreg](https://huggingface.co/Ray2333/GRM-Gemma-2B-sftreg)**(2B)** | 75.3    |   95.5  |  48.7 |   80.0 | 76.8     |  
|  [Gemma-2B-rewardmodel-baseline](https://huggingface.co/Ray2333/Gemma-2B-rewardmodel-baseline)**(2B)** | 73.7    |   94.1  |  46.1 |  79.6 |  75.0   |  




## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```

Then, go to the `scripts' folder and train the reward model with the default hyperparameters
```
cd scripts
sh train_bt_rm_full.sh
sh train_bt_rm_lora.sh
sh train_grm_full.sh
sh train_grm_lora.sh
```

Evaluating trained models on 'llm-blender/Unified-Feedback', 'HuggingFaceH4/hhh_alignment', 'lmsys/mt_bench_human_judgments':
```
sh eval_bt_rm.sh
sh eval_grm_rm.sh
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

## Acknowledgment
This repo is built upon [transformers](https://github.com/huggingface/transformers) and [trl](https://github.com/huggingface/trl), with also inspiration from [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling). 


