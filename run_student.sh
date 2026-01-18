#!/bin/bash
#SBATCH --job-name=student_input       # task name
#SBATCH --output=logs/%j.out         # 标准输出日志
#SBATCH --error=logs/%j.err          # 错误日志
#SBATCH --partition=NvidiaAll        # 分区名 (参考了你截图中的 srun 参数)
#SBATCH --nodes=1                    # 节点数
#SBATCH --ntasks=1                   # 任务数
#SBATCH --time=1440                  # 时间限制 (分钟)，参考了你的 -t 1440


source /home/r/ruan/miniconda3/etc/profile.d/conda.sh
conda activate bcsd

echo "Start running student_input.py..."
python ./src/student_input.py
echo "Job finished."