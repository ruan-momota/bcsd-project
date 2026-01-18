import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class DistillationDataset(Dataset):
    def __init__(self, student_pt_path, teacher_pt_path, train_projects=['clamav', 'curl', 'nmap']):
        print(f"Loading Student Data from {student_pt_path} ...")
        student_data = torch.load(student_pt_path)
        # student_data keys: 'student_input' (dict of tensors), 'meta_data' (list)
        
        print(f"Loading Teacher Data from {teacher_pt_path} ...")
        teacher_data = torch.load(teacher_pt_path) 
        # teacher_data structure: {'filename.json': tensor[N, 768]}
        
        self.inputs = [] # list of dicts (lightweight) or indices

        valid_indices = []
        aligned_teacher_vecs = []
        
        print("Aligning Student and Teacher Data...")
        
        # 辅助计数器，用于追踪当前文件在 teacher tensor 中的进度
        
        # 按文件名分组 student meta_data 以便和 teacher 匹配
        # 但 student data 已经是平铺的了，我们需要遍历 meta_data
        
        # 优化策略：先建立 teacher 数据的查找表
        # 检查 teacher_data 的 key 是否包含完整文件名
        
        # temp storage
        selected_input_ids = []
        selected_attn_masks = []
        selected_token_types = []
        selected_teacher_vecs = []
        
        # 遍历所有样本
        full_input_ids = student_data['student_input']['input_ids']
        full_attn_masks = student_data['student_input']['attention_mask']
        full_token_types = student_data['student_input']['token_type_ids']
        meta_list = student_data['meta_data']
        
        # 我们需要维护每个文件内部的 index 计数
        file_counters = {} 
        
        skipped_files = set()
        
        for global_idx, meta in enumerate(tqdm(meta_list, desc="Aligning")):
            project = meta['project']
            filename = meta['file_name']
            
            # filter project
            if project not in train_projects:
                continue
                
            # retrieve teacher embeds
            if filename not in teacher_data:
                if filename not in skipped_files:
                    # print(f"Warning: {filename} not found in teacher embeddings.")
                    skipped_files.add(filename)
                continue
                
            teacher_file_tensor = teacher_data[filename] # [N_funcs, 768]
            
            # 确定当前函数在该文件中的索引
            if filename not in file_counters:
                file_counters[filename] = 0
            
            local_idx = file_counters[filename]
            
            # 安全检查：防止索引越界 (比如 Student 处理了更多函数)
            if local_idx >= teacher_file_tensor.shape[0]:
                continue
                
            # 获取对应的 teacher vector
            t_vec = teacher_file_tensor[local_idx] # [768]
            
            selected_input_ids.append(full_input_ids[global_idx])
            selected_attn_masks.append(full_attn_masks[global_idx])
            selected_token_types.append(full_token_types[global_idx])
            selected_teacher_vecs.append(t_vec)
            
            file_counters[filename] += 1

        print(f"Alignment Done. Valid Training Samples: {len(selected_input_ids)}")
        
        if len(selected_input_ids) == 0:
            raise ValueError("No valid training samples found! Check paths and project names.")

        # convert to tensor
        self.input_ids = torch.stack(selected_input_ids)
        self.attention_mask = torch.stack(selected_attn_masks)
        self.token_type_ids = torch.stack(selected_token_types)       
        self.teacher_vecs = torch.stack(selected_teacher_vecs)
        
    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'teacher_embed': self.teacher_vecs[idx]
        }
            
        return item