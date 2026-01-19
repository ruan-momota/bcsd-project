#!/bin/bash
#SBATCH --job-name=base_1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1440

source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running train_baseline_1.py..."
python ./src/train_baseline_1.py
echo "Job finished."