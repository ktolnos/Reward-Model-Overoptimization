# Best-of-N (BoN)  [Under Preparation]

## Overview

For BoN experiment, we use a 20K subset of the Unified-Feedback dataset, labeled by a gold reward model, to train proxy reward models. BoN sampling is then conducted on a held-out 1K test set, where \( N \) responses are generated per prompt. The proxy model ranks these responses, selecting the top responses based on their proxy scores. The selected responses are then evaluated by the gold reward model, and average scores (both proxy and gold) across the test set reflect the quality of responses chosen by the proxy reward model.

Scripts for each step can be found in the `scripts/bon`.


## Experiment Steps

### Step 1: Obtain Gold Reward for Training Data

1. **Dataset Selection**: A 20K subset of the Unified-Feedback dataset is used for training, and a 1K test set is reserved for evaluation.
2. **Labeling**: Both training and test sets are labeled by the [gold reward model](https://huggingface.co/Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback), a 7B human preference model fine-tuned on the full Unified-Feedback dataset. This provides a gold standard against which the proxy models are compared.
3. **Labeling**: After obtaining the gold score, replace the original scores in dataset with the scores labeled by the gold model.

### Step 2: Train Proxy Reward Model

Using the gold reward-labeled training data, we train a proxy reward model to approximate the scoring behavior of the gold model. This model will later be used to score generated responses.

### Step 3: Generate Samples Using the Policy Model

For each prompt in the 1K test set, the policy model generates \( N \) responses (where \( N \) varies from 1 to 405). These responses represent a wide range of potential answers from which the proxy model will select the best candidate.

### Step 4: Obtain Proxy Scores for Generated Samples

The trained proxy model is applied to each of the generated responses, assigning a proxy reward score to each response. This score reflects the model’s evaluation of the response quality based on the proxy’s learned preferences.

### Step 5: Select Best-of-N Responses

Using the proxy scores from Step 4, we select the single best response out of the \( N \) generated responses for each prompt. The highest-scoring response is chosen as the “best-of-N” according to the proxy model.

### Step 6: Evaluate Selected Responses with Gold Reward Model

The selected best-of-N responses from Step 5 are then evaluated by the gold reward model to obtain a gold score. This step measures the true quality of the responses chosen by the proxy model in alignment with the gold standard.

### Step 7: Collect Results

For each \( N \) value (from 1 to 405), we calculate the average proxy and gold scores across all test prompts. These scores reflect the proxy model’s effectiveness at selecting responses that align with gold standards as \( N \) increases.
