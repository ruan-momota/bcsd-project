# experiment record

## dataset related

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

- valid funcs: 拥有>=2个变体的函数（同名函数，拥有不同编译器和优化等级的版本）

- train: nmap,openssl,unrar,zlib (962096)
- test: clamav, curl (142794)

- 脏数据：函数名不同，但是函数体相同。（目前依据相同的teacher向量来判断）
因为不同的函数名，triplet时可能作为负样本拉大之间的距离，但是相同的语义会产生矛盾。

- 去掉脏数据后：
Total Functions Scanned      : 3,772,046
Exact Duplicates (Ignored)   : 1,693,585 (Same name, same content)
Renamed Duplicates (DIRTY)   : 937,977 (Diff name, same content)
Dirty Data Rate              : 24.87%
------------------------------------------------------------
Example Conflicts (First 10 detected):
'_Z14UnixSlashToDosPKcPcm' vs '_Z14DosSlashToUnixPKcPcm' (in unrar)
'_Z14UnixSlashToDosPKwPwm' vs  '_Z14DosSlashToUnixPKwPwm' (in unrar)
'_ZN5ArrayIwE4SizeEv' vs '_ZN5ArrayIcE4SizeEv' (in unrar)
'_ZN5ArrayIwE9CleanDataEv' vs '_ZN5ArrayIcE9CleanDataEv' (in unrar)
'_ZNK5ArrayIcEixEm' vs  '_ZNK5ArrayIhEixEm' (in unrar)
------------------------------------------------------------
Top 10 Projects with Renamed Duplicates:
z3               : 794086 dirty functions
openssl          : 68039 dirty functions
nmap             : 51882 dirty functions
curl             : 17021 dirty functions
clamav           : 5349 dirty functions
unrar            : 1309 dirty functions
zlib             : 291 dirty functions

- 第二次排除z3后再去除脏数据，结果有差异：
Total Functions Scanned      : 1,104,890
Exact Duplicates (Ignored)   : 584,323 (Same name, same content)
Renamed Duplicates (DIRTY)   : 141,220 (Diff name, same content)
Dirty Data Rate              : 12.78%
------------------------------------------------------------
Top 10 Projects with Renamed Duplicates:
openssl            : 65676 dirty functions
nmap               : 51882 dirty functions
curl               : 17001 dirty functions
clamav             : 5080 dirty functions
unrar              : 1309 dirty functions
zlib               : 272 dirty functions

疑问：
- 是否应该进一步去掉脏数据，比如原始汇编只有两行，都是extern

## memory related

teacher_gen.py: 
- tokenizer_len=128, batch=256: 3272MiB /  8192MiB

length 512, batch 256 oom
length 512, batch 128 oom
length 512, batch 64 OK
length 1024, batch 128 oom
length 1024, batch 64 oom
length 1024, batch 32 OK

train_distill.py:
- batch = 256: OOM
- batch = 128: 4156MiB / 8192MiB

## benchmark related

benchmark:
1000 queries, 100000 pool

benchmark（包含z3）:
- 1000 queries, 100000 pool
- test: z3(2667156), nmap(526430), total 3193586
- diff funcs(group): 122409
- 结果1000个query全部是z3

benchmark_1:
- Loading Test Projects from pre-tokenized inputs: ['clamav', 'curl']
    - Loaded clamav: 88173 functions                       
    - Loaded curl: 54621 functions         
- Total functions in pool: 142794
- Unique function names with >= 2 variants: 7551

问题：
- 应该分为训练集，验证集，测试集，而不是只有训练集和测试集

## model related

疑问：
- 模型config是否合理：
max_position_embeddings=512,
type_vocab_size=512

## training related

- 未去除脏数据
71778 base
71779 distil
71985 distil+triplet

- 去除脏数据后，不包含z3：
baseline
    - MRR@10 = 0.5083, Recall@1 = 0.5000
distillation
    - MRR@10 = 0.5070, Recall@1 = 0.5000
distillation + triplet
    - MRR@10 = 0.5070, Recall@1 = 0.4990

问题：
- baseline 随机抽取样本的时候需不需要固定种子

## teacher related

先想办法提高teacher的准确率

-teacher 512: 73341
MRR: 0.5091，Recall: 0.5020
- teacher 1024: 73665
MRR: 0.5078，Recall: 0.4990


## new

- 使用256 teacher，不算z3
去除脏数据后（同名或不同名，函数体相同的）

Global Statistics:
  Total Processed:    1104890
  Kept (Unique):      409161
  Blocked (Dirty):    135051 (Same Body, Diff Name)
  Blocked (Exact):    560678 (Same Body, Same Name)
  Total Filtered Rate: 62.97%
----------------------------------------
Kept Functions by Project (Descending):
  openssl             : 183547
  nmap                : 137563
  clamav              : 47491
  curl                : 21382
  unrar               : 14510
  zlib                : 4668

- 使用128 teacher，算z3
去除脏数据后（同名或不同名，函数体相同的）

训练：openssl, clamav, zlib, nmap (373269)
验证：unrar (14510)
测试：curl (21382)

bcsd
=== Building VAL Set (['unrar']) ===
  - Total available functions: 14510                                                                                                                             
  - Unique function names with >= 2 variants: 818
  - Constructing Query Set and Gallery Pool...
  - [Note] Pool already larger than requested size due to variants (13601)
  - Saved val set: 818 Queries, 13601 Pool size.

=== Building TEST Set (['curl']) ===
  - Total available functions: 21382                                                                                                                             
  - Unique function names with >= 2 variants: 1168
  - Constructing Query Set and Gallery Pool...
  - [Note] Pool already larger than requested size due to variants (17277)
  - Saved test set: 1000 Queries, 17277 Pool size.

- baseline
Total samples loaded: 373269
Skipped 648842 dirty samples based on blocklist.
Valid function groups (>=2 variants): 28282

- distill
Total Scanned Samples: 1022111
Skipped 648842 dirty samples based on blocklist.
Total Clean Samples Loaded: 373269
Initializing Student Model...

- mlm
Total samples for MLM: 373269

(bcsd) ruan@kaolin:~/binai/bcsd-project (130) [0:35:17] % sbatch run_baseline.sh 
Submitted batch job 76795  128batch OOM 
(bcsd) ruan@kaolin:~/binai/bcsd-project (0) [0:36:11] % sbatch run_distill.sh 
Submitted batch job 76796 128batch OOM
(bcsd) ruan@kaolin:~/binai/bcsd-project (0) [0:36:17] % sbatch run_mlm.sh    
Submitted batch job 76797 64 batch OOM

(base) ruan@diamant:~/binai/bcsd-project (0) [10:06:12] % sbatch run_distill.sh 
Submitted batch job 76859
(base) ruan@diamant:~/binai/bcsd-project (0) [10:06:33] % sbatch run_baseline.sh 
Submitted batch job 76860

(bcsd) ruan@kaolin:~/binai/bcsd-project (0) [10:23:17] % sbatch run_distill_triplet.sh 
Submitted batch job 76874

76958 NvidiaAll     base     ruan PD       0:00      1 (Priority)
             76960 NvidiaAll    disti     ruan PD       0:00      1 (Priority)

             Teacher Precomputed Benchmark Result:
MRR    : 0.9607
Recall : 0.9440

## up to date

78042 NvidiaAll  tea2565     ruan PD       0:00      1 (Priority)
78043 NvidiaAll  stu2565     ruan PD       0:00      1 (Priority)