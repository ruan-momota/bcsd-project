# running record

total func: 3772046

## 1.17

change tokenizer.model_max_length to 128

teacher and student same tokenizer_len?

job: 71524
teacher_gen.py
tokenizer_len=128
batch=256
3272MiB /   8192MiB

## 1.18

benchmark:
- 1000 queries, 100000 pool
- test: z3(2667156), nmap(526430), total 3193586
- diff funcs(group): 122409
- 结果1000个query全部是z3

model batch 512: job 71635
