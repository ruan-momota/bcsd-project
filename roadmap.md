# BCSD知识迁移

目标：知识迁移，采用知识蒸馏，将sota的teacher model(CLAP-ASM)的高维语义知识蒸馏到一个轻量级的student model中。

方法：对比baseline model和student model在BCSD任务上的表现。

预估：student model的指标（MRR10）优于Baseline model，验证知识迁移的有效性。

## 1 数据预处理和知识准备

### 1.1 二进制转汇编

- 使用IDA Pro和CLAP的process_asm.py在本地将二进制文件转换成rebase后的汇编
    - 每个二进制文件对应一个json格式的汇编文件，格式为[{},{}...]，每个元素对应一个函数
    - process_asm.py的rebase将跳转指令后的loc_替换为INSTR加行号的相对地址

- 先针对x64平台的二进制文件

### 1.2 student输入准备

目标：将汇编转换为Student模型的Token ID列表

- 基于1.1生成的json格式汇编，直接使用teacher模型CLAP-ASM的tokenizer生成input

- 检查生成的input，输出小部分内容到本地进行检查

产出：由json格式汇编由tokenizer转换成的input，作为student和baseline模型的输入

### 1.3 teacher知识向量生成

目标：利用CLAP-ASM模型为每个函数生成一个teacher语义向量

- 从Hugging Face加载CLAP-ASM

- 将1.1生成的json格式的汇编利用CLAP-ASM的tokenizer生成对应的input

- 利用teacher模型CLAP-ASM的encoder将input生成对应的知识向量

产出：知识矩阵teacher_target，作为知识蒸馏的目标

## 2 模型架构定义

目标：定义一个统一的轻量级Bert架构，同时用于Baseline和student model
- 要求baseline和student模型在rtx2060 12gb上能一个小时内训练完，调整合适的层数和维度
- Embedding层：将Token ID映射为256维向量
- Transformer Encoder：4层Transformer Blocks，8个Attention Heads，隐藏层维度256（暂定，可调整）
- Pooling层：获取[CLS]Token的输出向量，作为整个函数的嵌入

产出：一个Small Bert模型

## 3 数据集划分和评估

目标：建立训练、验证和测试集，建立BCSD评估框架

### 3.1 数据集划分

- 按项目划分

- 训练集：clamav,curl,nmap
- 测试集：zlib,openssl

### 3.2 BCSD评估

- 使用测试集

- 定义正样本：比如给定义一个函数，正样本是其他编译器，优化等级下的该函数
- 定义负样本：所有非该函数

- 随机选择1000个函数作为查询集

- 随机选取10000个函数构建candidate pool

- 生成Ground Truth（json文件），映射每个Query在Pool中的正确候选项。

- 评估函数：计算Query和Pool向量的余弦相似度，并对照Ground Truth计算MRR10和Recall@1

## 4 Baseline的训练和评估

目标：建立无知识迁移的，仅使用TripletLoss从零开始训练的SmallBERT模型的性能基线。

### 4.1 构建Baseline

- 实例化SmallBERT模型，使用随机初始化权重

### 4.2 训练Baseline

- 输入：从训练集动态采样三元组（Anchor, Positive, Negative）。（输入是三个操作码ID序列）

- 过程：SmallBERT分别计算三个序列的[CLS]嵌入向量

- 损失函数：TripletLoss，使(Anchor, Positive)距离最小化，（Anchor, Nagative）距离最大化

### 4.3 评估Baseline

- 获取Baseline的MRR10和Recall@1

## 5 student model知识蒸馏和微调

目标：验证KD能否让相同架构的SmallBERT达到更高性能。

### 5.1 KD预训练

目标：使SmallBERT学会用操作码序列复现Teacher(CLAP-ASM)的高维语义向量

- 同样实例化模型

- 输入：Dataset-1中的一个Batch的操作码ID序列

- Teacher推理：输入汇编文本 -> 得到N个向量 -> 计算NxN相似度矩阵S_teacher

- Student推理：输入Opcodes IDs -> 得到N个向量 -> 计算NxN相似度矩阵S_student

- 损失函数：Loss = KL_Divergence(S_teacher, S_Student)

- 产出：一个预训练好的Student Model


### 5.2 模型微调

目标：在KD预训练基础上，针对BCSD任务调优

- 采用与Baseline相同的TriletLoss训练方案。

产出：微调后的Student Model

- 优化：将微调和KD预训练结合，Loss_total = α*Loss_KD + β * Loss_triplet

### 5.3 模型评估

- 使用微调后的Student Model，在测试集上运行BCSD评估。

产出：student model的MRR10和Recall@1

## 6 分析

- 对比Baseline和Student Model的MRR10和Recall@1

- 期望：Student_MRR10 > Baseline_MRR10


