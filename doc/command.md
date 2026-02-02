srun -N 1 -p NvidiaAll -t 1440 -w dacit --pty zsh

srun --jobid=71686 nvidia-smi

squeue --job <你的JOBID> --start