# Agenda

1. Motivation
2. Experimental Setup
    - Data Preparation
    - Model Architecture
    - Dataset & Metrics
3. The Baseline
4. Knowledge Distillation
5. Fine-tuning
6. Evaluation & Analysis

## 1 Motivation

本项目致力于通过 Ranking-based 知识蒸馏，将 CLAP-ASM 的 768 维高阶语义降维并注入到 256 维的轻量化SmallBERT 中，以此验证知识迁移在提升小模型BCSD任务指标MRR10和recall1上的有效性。

## 2 Exprtimental Setup

- 数据准备： 我们聚焦于 x64 架构下的二进制文件。我们利用 IDA Pro 提取汇编代码，并使用 CLAP 的 process_asm.py 脚本将其标准化为 JSON 格式，构建了我们的原始语料库。
- 模型架构：
Teacher Model: 我们选用 CLAP-ASM 作为教师模型。
Student/Baseline Model: 我们设计了一个轻量级的 SmallBERT。它仅有 4 层隐藏层，隐藏层维度为 256，配备 8 个注意力头。这不仅作为我们的学生模型，也是后续对比的 Baseline。
- 数据集划分：
我们选取了 openssl, clamav, nmap 以及 zlib 的部分数据作为训练集。
使用 unrar 作为验证集。
使用 curl 作为测试集。
- 数据清洗： 为了保证 Triplet Loss 训练的有效性，我们严格过滤了两类“脏数据”：
重复样本： 即函数名和函数体完全相同的冗余数据。
同体异名： 即函数体二进制完全相同，但函数名不同的情况（通常由编译优化或别名产生）。
最终，清洗后的训练集包含约 37 万样本，验证集 1.4 万，测试集 2.1 万。
- 评估指标： 我们构建了 BCSD Benchmark。在测试集中，我们设置了 1000 个 Query 和 17277 个 Pool size，通过 MRR10 和 Recall@1 来量化评估模型性能。"

## 3 The Baseline

- 采样策略： Baseline 采用 Triplet 采样。
Anchor & Positive: 随机选择一对‘函数名相同、但函数体不同’的函数（模拟不同编译器/优化等级下的同一函数）。
Negative: 随机选择一个不同函数名的函数。
- 训练配置： 我们使用 Triplet Loss 作为损失函数。每个 Epoch 处理 60 万个样本，总共训练 20 个 Epoch。

## 4 Knowledge Distillation

- 损失函数： 与 Baseline 不同，这里我们将 Teacher (CLAP) 和 Student (SmallBERT) 的相似度矩阵作为输入。我们利用 KL 散度 (KL Divergence) 来约束 Student。目的是让 Student 在一个 Batch 内，学习 Teacher 对样本相似度的排序分布，而不仅仅是点对点的逼近。
- 训练规模： 每个 Epoch 包含 37 万个样本，同样训练 20 个 Epoch。

## 5 Fine-tuning

我们直接加载上一阶段蒸馏好的 Student Model 权重。
使用与 Baseline 相同的 Triplet Loss 进行微调。这一步的目的是将蒸馏学到的泛化语义，重新聚焦到区分正负样本的度量空间上。"

## 6 Evaluation & Analyze

总体表现： 从 BCSD Benchmark 的结果来看，我们观察到了明显的性能阶梯：Fine-tuned Student > Distilled Student > Baseline。
- 可以基于数据图进行深入分析。比如样本数量的影响，tokenizer长度的影响, baseline尝试使用MLM等等，我也不知道。

关于收敛速度 (Convergence):

 "值得注意的是，经过蒸馏的模型在 Fine-tuning 阶段的收敛速度通常比 Baseline 从头训练要快得多，因为它已经具备了良好的参数初始化。"

关于维度压缩 (Efficiency):

"虽然 Student 模型的维度仅为 Teacher 的 1/3 (256 vs 768)，但最终指标却保持了较高的水准，这证明了我们的轻量化策略在实际部署中的潜力。"