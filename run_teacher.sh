#!/bin/bash
#SBATCH --job-name=teacher_gen_256
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=NvidiaAll
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2880

source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running teacher_gen.py..."
python ./src/teacher_gen.py
echo "Job finished."