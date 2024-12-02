from datasets import load_dataset
from utils import create_output_directory, save_results_in_parquet_splits

        

data_path = "llm-blender/Unified-Feedback"
# Load the dataset
ds = load_dataset(data_path, 'all', split="train")
# Shuffle the dataset
ds_shuffled = ds.shuffle(seed=42)  # Using a seed for reproducibility
# Select the first 20,000 samples for the larger subset
subset_20k = ds_shuffled.select(range(20000))
# Select the next 2,000 samples for the smaller subset
subset_2k = ds_shuffled.select(range(20000, 21000))

# Now subset_20k contains 20,000 unique samples and subset_1k contains 1,000 unique samples
save_results_in_parquet_splits(subset_20k, num_splits=2, save_path="./data/unified_20k", mode='train')
save_results_in_parquet_splits(subset_2k, num_splits=1, save_path="./data/unified_1k", mode='test')