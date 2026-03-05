srun -N 1 -p NvidiaAll -t 1440 -w dacit --pty zsh
srun -N 1 -p AMD -t 1440 -w kaolin --pty zsh

srun --jobid=71686 nvidia-smi

squeue --job <你的JOBID> --start

!python train.py 2>&1 | tee output_log.txt

nohup python -u ./src/train_baseline.py > baseline.txt 2>&1 &

pip install -r requirements.txt