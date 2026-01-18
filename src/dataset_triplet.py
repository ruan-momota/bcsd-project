import torch
from torch.utils.data import Dataset
import random
import os
from collections import defaultdict
import config
from tqdm import tqdm

class BCSDTripletDataset(Dataset):
    def __init__(self, projects, max_length=128, epoch_sample_rate=40):
        self.max_length = max_length
        self.projects = projects
        self.epoch_sample_rate = epoch_sample_rate
        self.base_dir = os.path.join(config.DATA_DIR, "outputs", "student", str(self.max_length))
        
        self.samples = []                   
        self.groups = defaultdict(list)     # func_name -> list of indices in self.samples

        print(f"Loading pre-tokenized data for projects: {self.projects}")
        
        total_files = 0
        
        for proj_name in tqdm(self.projects, desc="Loading Projects"):
            proj_path = os.path.join(self.base_dir, proj_name)
            
            if not os.path.exists(proj_path):
                print(f"[Warning] Project directory not found: {proj_path}. Skipping.")
                continue
            
            pt_files = sorted([f for f in os.listdir(proj_path) if f.endswith('.pt')])
            
            if not pt_files:
                print(f"[Warning] No .pt files found in {proj_path}.")
                continue

            for pt_file in pt_files:
                file_path = os.path.join(proj_path, pt_file)
                try:
                    chunk_data = torch.load(file_path)
                    
                    start_idx = len(self.samples)
                    self.samples.extend(chunk_data)
                    
                    for i, item in enumerate(chunk_data):
                        global_idx = start_idx + i
                        func_name = item['func_name']
                        self.groups[func_name].append(global_idx)
                        
                    total_files += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        # filter func with one variance
        self.valid_func_names = [k for k, v in self.groups.items() if len(v) >= 2]
        
        print(f"Processed {total_files} .pt files.")
        print(f"Total samples loaded: {len(self.samples)}")
        print(f"Valid function groups (>=2 variants): {len(self.valid_func_names)}")
        
    def __len__(self):
        # number of samples per one epoch
        return len(self.valid_func_names) * self.epoch_sample_rate

    def __getitem__(self, idx):
        group_idx = idx % len(self.valid_func_names)
        group_name = self.valid_func_names[group_idx]
        group_indices = self.groups[group_name]
        
        if len(group_indices) < 2:
            raise ValueError(f"Function '{group_name}' has less than 2 samples.")
        
        anc_idx, pos_idx = random.sample(group_indices, 2)

        # choose one neg randomly
        while True:
            neg_name = random.choice(self.valid_func_names)
            if neg_name != group_name:
                neg_indices = self.groups[neg_name]
                neg_idx = random.choice(neg_indices)
                break
                
        item_anc = self.samples[anc_idx]['student_input']
        item_pos = self.samples[pos_idx]['student_input']
        item_neg = self.samples[neg_idx]['student_input']

        input_ids = torch.stack([
            item_anc['input_ids'], 
            item_pos['input_ids'], 
            item_neg['input_ids']
        ])
        
        attention_mask = torch.stack([
            item_anc['attention_mask'], 
            item_pos['attention_mask'], 
            item_neg['attention_mask']
        ])
        
        token_type_ids = torch.stack([
            item_anc['token_type_ids'], 
            item_pos['token_type_ids'], 
            item_neg['token_type_ids']
        ])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }