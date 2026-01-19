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

teacher embed全量加载用时：6h

benchmark:
- 1000 queries, 100000 pool
- test: z3(2667156), nmap(526430), total 3193586
- diff funcs(group): 122409
- 结果1000个query全部是z3

model batch 512: job 71683 out

注意模型以下值设置：
max_position_embeddings=512,
type_vocab_size=512,           

71684 distill

model batch 256: job 71685 out

128: 4156MiB /   8192MiB

Project Name         | Files    | Raw Funcs    | Valid Funcs 
------------------------------------------------------------
clamav               | 240      | 158127       | 88173                        
curl                 | 40       | 66797        | 54621       
nmap                 | 120      | 495539       | 390057                       
openssl              | 400      | 741787       | 526430                    
unrar                | 40       | 37316        | 28158       
z3                   | 40       | 2828183      | 2667156     
zlib                 | 160      | 27375        | 17451                       

------------------------------------------------------------
TOTAL                | 1040     | 4355124      | 3772046     

3772046-2667156=1104890
train:88w
test:22w

-------------------

train: nmap,openssl,unrar,zlib (95w)
test: clamav, curl (14w)

benchmark_1
- Loading Test Projects from pre-tokenized inputs: ['clamav', 'curl'] ...
    - Loaded clamav: 88173 functions                       
    - Loaded curl: 54621 functions         
- Total functions in pool: 142794
- Unique function names with >= 2 variants: 7551

dataset_triplet
- Processed 720 .pt files.
- Total samples loaded: 962096
- Valid function groups (>=2 variants): 26562

dataset_distill
- same as dataset_triplet

71726: baseline_1
71728: distill 1749/8192(128 batch，可以尝试512)
71766 baseline_1 4179MiB /   8192MiB

71778 base
71779 diti

baseline 随机抽取样本的时候需不需要固定种子

71985 d+t

脏数据过滤