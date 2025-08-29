#!/bin/bash
#SBATCH --job-name=ccnEG
#SBATCH --time=00:15:00
#SBATCH --array=1-2340
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/tcrv4423/slurm_out/slurm_output_%A_%a.out  


module load miniforge
conda activate hpc

# Define arrays
seeds=(42 22 1)
datasets=("10")
lrs=(1e-11 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1)
dlrs=(0 1e-11 1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1)
decays=(1e-10 1e-9 1e-8 1e-7 0)


# Base directory for outputs
folder="cnnEG"
BASE_DIR="/scratch/tcrv4423/${folder}"
OUT_DIR="${BASE_DIR}/output"
ERR_DIR="${BASE_DIR}/error"

# Ensure directories exist
mkdir -p "$OUT_DIR" "$ERR_DIR"

combo_id=0
for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    for lr in "${lrs[@]}"; do
      for dlr in "${dlrs[@]}"; do
        for decay in "${decays[@]}"; do
          combo_id=$((combo_id + 1))
          if [ "$combo_id" -eq "$SLURM_ARRAY_TASK_ID" ]; then
            
            # Format values safely
            safe_lr=$(echo "$lr" | sed 's/e/E/g')
            safe_dlr=$(echo "$dlr" | sed 's/e/E/g')
            safe_decay=$(echo "$decay" | sed 's/e/E/g')
  
            # Choose prefix based on dlr
            if [ "$dlr" = "0" ]; then
              prefix="cor"
              outname="${prefix}_${dataset}_${safe_lr}__${safe_dlr}_${safe_decay}_${seed}"
            else
              prefix="decor"
              outname="${prefix}_${dataset}_${safe_lr}_${safe_dlr}_${safe_decay}_${seed}"
            fi
  
            echo "Running combo $combo_id: $outname"
  
            python launchcnn.py \
              --dataset "$dataset" --lr "$lr" --dlr "$dlr" --decay "$decay" --seed "$seed" --folder "$folder" --EG "true" --bias "false" --wandb "MSc CNN EG" \
              > "${OUT_DIR}/${outname}.out" \
              2> "${ERR_DIR}/${outname}.err"
  
            exit 0
          fi
        done
      done
    done
  done
done

echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
exit 1
