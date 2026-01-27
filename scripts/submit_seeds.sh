#!/bin/bash

# Default values
START_SEED=${1:-400}
END_SEED=${2:-409}
SAVE_LAST_ONLY=${3:-True}
SKIP_OPTIMIZER=${4:-True}

echo "Submitting jobs for seeds $START_SEED to $END_SEED"
echo "Options: SAVE_LAST_ONLY=$SAVE_LAST_ONLY, SKIP_OPTIMIZER=$SKIP_OPTIMIZER"

for seed in $(seq $START_SEED $END_SEED); do
    echo "Submitting seed $seed"
    sbatch my_train_bt_rm_full.sh $seed $SAVE_LAST_ONLY $SKIP_OPTIMIZER
    sleep 1 # Avoiding too many submissions at once
done
