from datasets import load_dataset, DatasetDict
import huggingface_hub

# 1. Define dataset names
original_dataset_name = "ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B"
new_dataset_name = "ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B-10k"
random_seed = 42

# 2. Download the dataset
print(f"Downloading dataset: {original_dataset_name}")
dataset = load_dataset(original_dataset_name)

# 3. Shuffle and select 25% of each split
processed_splits = {}
for split_name, split_data in dataset.items():
    print(f"Processing split: {split_name}")
    shuffled_split = split_data.shuffle(seed=random_seed)
    selected_split = shuffled_split.select(range(int(len(shuffled_split) * 0.25)))
    processed_splits[split_name] = selected_split

processed_dataset = DatasetDict(processed_splits)

# 4. Upload the new dataset
print(f"Uploading dataset to: {new_dataset_name}")
processed_dataset.push_to_hub(new_dataset_name)

print("Script finished successfully!")
