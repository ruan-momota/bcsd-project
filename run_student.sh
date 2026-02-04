#!/bin/bash
#SBATCH --job-name=stu2565     
#SBATCH --output=logs/%j.out       
#SBATCH --error=logs/%j.err       
#SBATCH --partition=NvidiaAll       
#SBATCH --nodes=1              
#SBATCH --ntasks=1                   
#SBATCH --time=1440                


source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running student_input.py..."
python ./src/student_input.py
echo "Job finished."