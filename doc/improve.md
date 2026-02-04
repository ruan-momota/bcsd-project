注意点：

重点讲述interesting，少讲比如数据集具体数量

数据多用图片



-----------

slide motivation: 介绍clap-asm
指标选择更改，因为每个样本会有多个variants正确答案，但是MRR10和recall1只统计第一个
slide setup: 去除的脏数据具体例子
baseline: 将loss函数可视化
distillation：将loss函数可视化

evaluation实际上是结论，缺乏具体的分析
重点关注不同的模型的变体之间的比较（明确不同的变体）
- baseline(tripletloss, MLM)
- distillation(KD, ...)
- fine-tuning(student based)