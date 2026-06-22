if [ "$SLURM_CLUSTER_NAME" == "killarney" ]; then
    SBATCH_ARGS="--gres=gpu:l40s:1 -A aip-rahulgk --qos=normal --nodelist="
    module load cuda/12.9
    source ${scratch_dir}/venv/rmo/bin/activate
    export WANDB_PROJECT=RMO
else
    SBATCH_ARGS=""
fi

sbatch $SBATCH_ARGS $1