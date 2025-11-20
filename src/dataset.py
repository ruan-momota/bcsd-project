# src/dataset.py
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import config
import os

class OpcodeTokenizer:
    def __init__(self, vocab_path=config.VOCAB_FILE):
        # 检查词汇表是否存在
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"词汇表未找到: {vocab_path}。请先运行 preprocess.py。")
            
        self.vocab = self.load_vocab(vocab_path)
        # 创建反向词汇表 (id -> word)，用于调试或解码
        self.id2word = {v: k for k, v in self.vocab.items()}
        
        self.unk_id = self.vocab.get(config.UNK_TOKEN)
        self.pad_id = self.vocab.get(config.PAD_TOKEN)
        self.cls_id = self.vocab.get(config.CLS_TOKEN)
        self.sep_id = self.vocab.get(config.SEP_TOKEN)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def encode(self, tokens, max_length=config.MAX_SEQ_LEN):
        """
        将token文本列表转换为ID列表，并处理 Padding/Truncation
        返回: (input_ids, attention_mask)
        """
        # 1. 映射为 ID，如果不在词表中则使用 UNK
        token_ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        
        # 2. 添加 [CLS] 和 [SEP]
        # 结构: [CLS] ...tokens... [SEP]
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        
        # 3. 计算真实长度 (用于生成 attention_mask)
        actual_len = len(token_ids)
        
        # 4. 截断或填充
        if actual_len > max_length:
            # 截断：保留 [CLS]，截断中间，最后保留 [SEP] 是通常做法
            # 这里为了保持上下文连贯，简单截断末尾
            token_ids = token_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            # 填充
            pad_len = max_length - actual_len
            token_ids = token_ids + [self.pad_id] * pad_len
            # Mask: 真实token位置为1，PAD位置为0
            attention_mask = [1] * actual_len + [0] * pad_len
            
        return token_ids, attention_mask

    def get_vocab_size(self):
        return len(self.vocab)

class BCSDDataset(Dataset):
    """
    PyTorch Dataset 类，用于训练时加载已处理好的 ID 数据
    """
    def __init__(self, data_path=config.TRAIN_ID_FILE):
        self.data = []
        print(f"Loading dataset from {data_path}...")
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"ID数据文件未找到: {data_path}。请先运行本脚本的 process_data_to_ids 函数。")

        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"数据集加载完毕，共 {len(self.data)} 条样本。")
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            "input_ids": torch.tensor(item['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(item['attention_mask'], dtype=torch.long),
            "func_name": item['func_name'],
            "project": item['project']
            # binary 名等其他元数据按需添加
        }

def process_data_to_ids():
    """
    一次性脚本：读取包含 'instructions' 的数据，
    将其拆解(Flatten)为 tokens，转化为 id 形式并保存。
    """
    print("开始序列化数据 (Instructions -> Tokens -> IDs)...")
    
    # 1. 初始化 Tokenizer
    try:
        tokenizer = OpcodeTokenizer()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Tokenizer 已加载，词汇表大小: {tokenizer.get_vocab_size()}")
    
    input_path = config.TRAIN_DATA_FILE
    output_path = config.TRAIN_ID_FILE
    
    if not os.path.exists(input_path):
        print(f"错误: 未找到输入文件 {input_path}。请先运行 preprocess.py")
        return

    count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        # 逐行读取，处理，写入
        for line in tqdm(f_in, desc="Processing"):
            item = json.loads(line)
            
            # --- 核心修改部分 Start ---
            # 旧逻辑: raw_tokens = item['tokens']
            # 新逻辑: 读取 'instructions' 列表并拆分为 token 列表
            # 例子: ["MOV EAX MEM", "RET"] -> ["MOV", "EAX", "MEM", "RET"]
            
            if 'instructions' in item:
                raw_tokens = []
                for insn_str in item['instructions']:
                    # 使用 split() 默认按空格分割
                    raw_tokens.extend(insn_str.split())
            elif 'tokens' in item:
                # 兼容旧格式（防止报错，但建议重新预处理）
                raw_tokens = item['tokens']
            else:
                # 数据异常跳过
                continue
            # --- 核心修改部分 End ---
            
            # 编码转换
            input_ids, attention_mask = tokenizer.encode(raw_tokens)
            
            # 构建新的记录 (只保留训练需要的最小字段以节省空间)
            new_item = {
                "func_name": item['func_name'],
                "project": item['project'],
                "binary": item.get('binary', ''), # 使用 get 防止旧数据缺失 key
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            f_out.write(json.dumps(new_item) + "\n")
            count += 1
            
    print(f"处理完成！已生成 {count} 条ID数据。")
    print(f"保存位置: {output_path}")

if __name__ == "__main__":
    # 直接运行此文件即可生成 ID 数据集
    process_data_to_ids()