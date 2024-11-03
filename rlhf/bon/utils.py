import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import torch


def create_output_directory(log_dir: str, wandb_name: str):
    output_path = os.path.join(log_dir, wandb_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


# Function to save results as Parquet files
def save_results_in_parquet_splits(results, num_splits, save_path, mode='test'):
    results_df = pd.DataFrame(results)
    dataset_with_results = Dataset.from_pandas(results_df)
    
    split_size = len(dataset_with_results) // num_splits
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(dataset_with_results)
        split = dataset_with_results.select(range(start, end))
        file_path = f"{save_path}/{mode}-0000{i}-of-0000{num_splits}.parquet"
        split.to_parquet(file_path)


# Define the KL equation
def kl_equation(N):
    return np.log(N) - (N - 1) / N


# Calculate and filter KL values
def calculate_kl_values(N_values, kl_min=0, kl_max=5):
    kl_values = [kl_equation(N) for N in N_values]
    results = pd.DataFrame({'N': N_values, 'kl': kl_values})
    return results[(results['kl'] >= kl_min) & (results['kl'] <= kl_max)]


# Define function to get highest rewards within N items per group
def get_highest_within_n(group, n):
    return group.head(n).nlargest(1, 'rewards')