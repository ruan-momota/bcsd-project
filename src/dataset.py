import torch
from torch.utils.data import Dataset
import random
import os
from collections import defaultdict
from extract_asm import load_project_functions
import config

class BCSDTripletDataset(Dataset):
    def __init__(self, tokenizer, projects=['clamav', 'curl', 'nmap', 'unrar'], max_length=512):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = os.path.join(config.DATA_DIR, "asm_x64")
        
        self.samples = []                   # stored all funcions and related information
        self.groups = defaultdict(list)     # func_name -> list of indices in self.samples
        
        # load all training data
        print(f"Loading training data from: {projects}...")
        for proj in projects:
            proj_path = os.path.join(self.data_dir, proj)
            funcs = load_project_functions(proj_path) # all funcs in one json file
            
            start_idx = len(self.samples)
            self.samples.extend(funcs)
            
            for i, item in enumerate(funcs):
                global_idx = start_idx + i
                self.groups[item['func_name']].append(global_idx) # e.g. {"func1": [1, 99]}
                
        # filter functions that have more than one positive variants
        self.valid_func_names = [k for k, v in self.groups.items() if len(v) >= 2]
        print(f"Total functions: {len(self.samples)}")
        print(f"Valid groups: {len(self.valid_func_names)}")
        
    def __len__(self):
        # iteration time per epoch
        return len(self.valid_func_names)

    # use for DataLoader
    def __getitem__(self, idx):
        # choose an anchor func_name randomly
        group_name = self.valid_func_names[idx]
        group_indices = self.groups[group_name]
        
        # choose anchor and positive randomly
        # replace=False to ensure no duplication
        anc_idx, pos_idx = random.sample(group_indices, 2)
        
        # choose one nagetive randomly
        while True:
            neg_name = random.choice(self.valid_func_names)
            if neg_name != group_name:
                neg_indices = self.groups[neg_name]
                neg_idx = random.choice(neg_indices)
                break
                
        text_anchor = self.samples[anc_idx]['asm_text']
        text_pos = self.samples[pos_idx]['asm_text']
        text_neg = self.samples[neg_idx]['asm_text']
        
        # tokenize
        self.tokenizer.model_max_length = 512
        inputs = self.tokenizer(
            [text_anchor, text_pos, text_neg],
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return inputs   # shape of input_ids: [3, 512]