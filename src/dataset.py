import json
import torch
from torch.utils.data import Dataset
import config

class OpcodeTokenizer:
    def __init__(self, vocab_path=config.VOCAB_FILE):
        self.vocab = self.load_vocab(vocab_path)
        self.unk_id = self.vocab.get(config.UNK_TOKEN)
        self.pad_id = self.vocab.get(config.PAD_TOKEN)
        self.cls_id = self.vocab.get(config.CLS_TOKEN)
        self.sep_id = self.vocab.get(config.SEP_TOKEN)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def encode(self, tokens, max_length=config.MAX_SEQ_LEN):
        """
        将token列表转换为ID列表，并进行截断/填充
        """
        # 1. Convert to IDs
        token_ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        
        # 2. Add CLS and SEP (类似BERT的标准做法)
        # 格式: [CLS] token_ids [SEP]
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        
        # 3. Truncate or Pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [self.pad_id] * pad_len
            
        return token_ids

    def get_vocab_size(self):
        return len(self.vocab)

# 为了测试Dataset，我们可以先定义一个简单的加载器
class BCSDDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        
        # 编码
        input_ids = self.tokenizer.encode(tokens)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "func_name": item['func_name'],
            "project": item['project']
        }