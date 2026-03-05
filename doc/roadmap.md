# BCSD知识迁移

目标：知识迁移，采用知识蒸馏，将sota的teacher model(CLAP-ASM)的高维语义知识蒸馏到一个轻量级的student model中。
方法：对比baseline model和student model在BCSD任务上的表现。
预估：student model的指标（MRR10）优于Baseline model，验证知识迁移的有效性。

## 1 数据预处理和知识准备

### 1.1 二进制转汇编

- 使用IDA Pro和CLAP的process_asm.py在本地将二进制文件转换成rebase后的汇编
    - 每个二进制文件对应一个json格式的汇编文件，格式为[{},{}...]，每个元素对应一个函数
    - process_asm.py有rebase算法将跳转指令后的loc_替换为INSTR加行号的相对地址
    - CLAP-ASM的clap_modeling.py会利用INSTR+行号作为tokenize的位置输入

- 针对x64平台的二进制文件

### 1.2 student输入准备

目标：将汇编转换为Student模型的Token ID列表

- 基于1.1生成的json格式汇编，直接使用teacher模型CLAP-ASM的tokenizer生成token id

产出：由json格式汇编由tokenizer转换成的token id，作为student和baseline模型的输入

### 1.3 teacher知识向量生成

目标：利用CLAP-ASM模型为每个函数生成一个teacher语义向量

- 从Hugging Face加载CLAP-ASM
- 将1.1生成的json格式的汇编利用CLAP-ASM的tokenizer生成对应的input(encoding)
- 利用teacher模型CLAP-ASM的encoder将input生成对应的embedding

产出：知识矩阵teacher_target，作为知识蒸馏的目标

## 2 模型架构定义

目标：定义一个统一的轻量级Bert架构，同时用于Baseline和student model

- Embedding层：将Token ID映射为256维向量
- Transformer Encoder：4层Transformer Blocks，8个Attention Heads，隐藏层维度256（暂定，可调整）
- Pooling层：获取[CLS]Token的输出向量，作为整个函数的embedding

产出：一个Small Bert模型

## 3 数据集划分和评估

目标：建立训练、验证和测试集，建立BCSD评估框架

### 3.1 数据集划分

- 按项目划分
- 训练集：openssl, clamav, zlib, nmap
- 验证集：unrar
- 测试集：curl

### 3.2 BCSD评估

- 使用测试集
- 定义正样本：比如给定一个函数，正样本是不同编译器或不同优化等级下的该函数
- 定义负样本：所有非该函数

- 随机选择1000个函数作为查询集，如果不满1000个就按最大可供查询的函数数量
- 随机选取10000个函数构建candidate pool（其中包括所有query的变体）
- 生成Ground Truth（json文件），映射每个Query在Pool中的正确候选项。
- 评估函数：计算Query和Pool向量的余弦相似度，并对照Ground Truth计算MAP@50，R-Precision和NDCG@50

## 4 Baseline的训练和评估

目标：建立无知识迁移的，仅使用TripletLoss从零开始训练的SmallBERT模型的性能基线。

### 4.1 构建Baseline

- 实例化SmallBERT模型，使用随机初始化权重

### 4.2 训练Baseline

- 输入：从训练集动态采样三元组（Anchor, Positive, Negative）。（输入是三个函数的encoding）
- 过程：SmallBERT分别计算三个序列的嵌入向量
- 损失函数：TripletLoss，使(Anchor, Positive)距离最小化，（Anchor, Nagative）距离最大化
- 训练20个epoch

### 4.3 评估Baseline

- 获取Baseline的MAP@50，R-Precision和NDCG@50

## 5 student model知识蒸馏

目标：验证KD能否让相同架构的SmallBERT达到更高性能。

### 5.1 KD预训练

目标：用SmallBERT模仿Teacher(CLAP-ASM)的对汇编代码相似度的排序

- 注意点：
    - student和teacher模型的encoding相同
    - SmallBERT输出256维embedding，CLAP-ASM输出768维embedding
    - 使用ranking based ditillation方法
    - 查阅论文，可以尝试多种ranking based distillaiton方法，并进行横向对比

目标：用SmallBERT模仿Teacher(CLAP-ASM)的对汇编代码相似度的排序

- 注意点：
    - student和teacher模型的encoding相同
    - SmallBERT输出256维embedding，CLAP-ASM输出768维embedding
    - 使用ranking based ditillation方法
    - 查阅论文，可以尝试多种ranking based distillaiton方法，并进行横向对比

方案：基于概率分布KL-Divergence

- 构建新的数据集Distillation Dataset（不使用baseline的triplet采样）
    - 输入：汇编json文件和teacher_embedding
    - 索引对齐：json文件中每个函数的index要和teacher_embedding对齐
    - _/_getitem__: 不返回(anc,pos,neg)，而是返回单个函数样本
        包括input_ids,attention_mask,token_type_ids,teacher_embed四个维度

- 损失函数设计
    - student相似度矩阵：输入batch(N)和encoding序列，student输出Nx256向量，计算NxN相似度矩阵
      teacher相似度矩阵：对于teacher,通过DataLoader从pt文件直接获取Nx768向量，计算相似度矩阵
    - 温度调节：引入超参数T，在进行softmax之前，将相似度矩阵除以T，(初始设为2.0)
    - 概率分布对齐：将矩阵每一行进行归一化。每一行表示每个样本作为Query，和Batch内其他样本的相似度分布。
    - P_stu=softmax(S_stu/T), P_tea=softmax(S_tea/T)
    - 损失函数：LossKD=T^2 x nn.KLDivLoss(P_stu, P_tea)
    - 混合损失（可选）：为了防止 Student 在学习“相对关系”时彻底丢失“正样本必须拉近”的绝对目标，可以保留一部分 Task Loss

- 训练流程
    - 加载SmallBERT
    - Input: 从DataLoader获取input_ids和teacher_vecs
    - Forward: student_vecs = model(input_ids...)
    - Loss: 用LossKD计算student_vecs和teacher_vecs之间的散度
    - Backward: 更新student参数

- 训练20个Epoch，在BCSD测试集验证，对比student的MAP@50，R-Precision和NDCG@50是否超过Baseline

### 5.2 模型微调

- 在student的模型基础上使用tripletloss进行微调
- 获取20个epoch后经过微调的student模型MAP@50，R-Precision和NDCG@50数据

## 6 分析

- 对比只使用Tripletloss的Baseline，经过KD的Student Model，经过KD和tripletloss微调的Student Model

- 期望：微调 > KD > Baseline


