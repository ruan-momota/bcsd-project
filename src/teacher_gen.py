# src/teacher_gen.py
import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import config

class TeacherInferenceDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        print(f"正在加载数据: {data_path}")
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'instructions' in item:
                    self.samples.append(item['instructions'])
                else:
                    pass 
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def generate_teacher_knowledge():
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 加载 Teacher 模型
    print(f"正在加载 Teacher 模型: {config.TEACHER_MODEL_ID} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.TEACHER_MODEL_ID, 
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            config.TEACHER_MODEL_ID, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"模型加载出错: {e}")
        return

    model.to(device)
    model.eval()

    # 3. 准备数据加载器
    dataset = TeacherInferenceDataset(config.TRAIN_DATA_FILE)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,   
        num_workers=2,   
        collate_fn=lambda x: x 
    )
    
    all_embeddings = []

    print(f"开始生成知识向量 (Batch Size: {config.BATCH_SIZE})...")
    
    # 4. 推理循环
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Distilling Knowledge"):
            # 手动截断并拼接
            formatted_batch = [
                {"instructions": "\n".join(func_ins[:256])} 
                for func_ins in batch_data
            ]

            # Tokenize
            try:
                inputs = tokenizer(
                    formatted_batch, 
                    padding=True, 
                    return_tensors="pt"
                )
            except Exception as token_e:
                print(f"\nTokenization Error: {token_e}")
                continue
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # 模型前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # --- 核心修复逻辑 Start ---
            # 1. 检查输出是否直接就是 Tensor
            if isinstance(outputs, torch.Tensor):
                # 如果是二维 (Batch, Hidden)，说明已经是 Pooling 后的向量了，直接用
                if outputs.dim() == 2:
                    emb = outputs
                # 如果是三维 (Batch, Seq, Hidden)，说明是序列，取第一个 [CLS]
                else:
                    emb = outputs[:, 0, :]
            
            # 2. 如果是标准 HuggingFace 对象
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :]
            
            # 3. 兜底：如果是 tuple
            elif isinstance(outputs, (list, tuple)):
                 # 通常第一个元素是 hidden states
                 emb = outputs[0][:, 0, :]
            else:
                print(f"未知输出类型: {type(outputs)}")
                continue
            # --- 核心修复逻辑 End ---
            
            all_embeddings.append(emb.cpu())

    # 5. 拼接并保存
    if len(all_embeddings) > 0:
        teacher_knowledge_matrix = torch.cat(all_embeddings, dim=0)
        print(f"知识生成完毕。矩阵形状: {teacher_knowledge_matrix.shape}")
        
        torch.save(teacher_knowledge_matrix, config.TEACHER_EMBEDDINGS_FILE)
        print(f"知识矩阵已保存至: {config.TEACHER_EMBEDDINGS_FILE}")
    else:
        print("警告：未生成任何向量，请检查数据源。")

if __name__ == "__main__":
    generate_teacher_knowledge()