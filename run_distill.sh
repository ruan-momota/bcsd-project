#!/bin/bash
#SBATCH --job-name=disti     
#SBATCH --output=logs/%j.out     
#SBATCH --error=logs/%j.err          
#SBATCH --partition=NvidiaAll        
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                  
#SBATCH --time=480


source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running train_distill.py..."
echo "clean, new benchmark, stu 256, tea 256"
python ./src/train_distill.py
echo "Job finished."