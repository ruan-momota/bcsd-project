import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class SmallBERTStudent(nn.Module):
    def __init__(self, 
                 vocab_size=30000, 
                 hidden_size=256, 
                 num_layers=4, 
                 teacher_dim=768):
        super().__init__()
        
        # 1. 定义小巧的 BERT 配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512
        )
        
        # 2. 骨干网络
        self.bert = BertModel(config)
        
        # 3. 投影头 (Projector)
        # 负责把 256维 -> 映射到 Teacher 的 768维
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, teacher_dim),
            nn.Tanh(),
            nn.Linear(teacher_dim, teacher_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取 [CLS] token 的向量 (batch, 256)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        
        # 投影到 Teacher 空间 (batch, 768)
        final_emb = self.projector(cls_emb)
        
        return final_emb