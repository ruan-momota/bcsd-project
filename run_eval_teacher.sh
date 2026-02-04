#!/bin/bash
#SBATCH --job-name=eval_tea
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1440

source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running eval_teacher.py..."
echo "teacher 1024 on benchmark_2"
python ./src/eval_teacher.py
echo "Job finished."