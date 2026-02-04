#!/bin/bash
#SBATCH --job-name=dis_tri    
#SBATCH --output=logs/%j.out     
#SBATCH --error=logs/%j.err          
#SBATCH --partition=NvidiaAll        
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                  
#SBATCH --time=960                 


source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running train_distill_triplet.py..."
python ./src/train_distill_triplet.py
echo "Job finished."