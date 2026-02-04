import torch
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import config

class BCSDMLMDataset(Dataset):
    def __init__(self, projects, blocklist_file=None):
        self.base_dir = os.path.join(config.DATA_DIR, "outputs", "student", "256_5")
        if blocklist_file is None:
            blocklist_file = os.path.join(config.DATA_DIR, "outputs", "blocklist256_5.json")
        self.blocklist = set()
        if os.path.exists(blocklist_file):
            with open(blocklist_file, 'r', encoding='utf-8') as f:
                self.blocklist = set(json.load(f))

        self.samples = []
        print(f"Loading data for MLM: {projects}")
        
        for proj_name in tqdm(projects, desc="Loading Projects"):
            proj_path = os.path.join(self.base_dir, proj_name)
            if not os.path.exists(proj_path): continue
            pt_files = sorted([f for f in os.listdir(proj_path) if f.endswith('.pt')])
            for pt_file in pt_files:
                try:
                    data = torch.load(os.path.join(proj_path, pt_file))
                    for item in data:
                        unique_key = f"{item['proj_name']}|{item['file_name']}|{item['func_name']}"
                        if unique_key in self.blocklist:
                            continue
                        self.samples.append(item['student_input'])
                except Exception as e:
                    print(f"Error {pt_file}: {e}")

        print(f"Total samples for MLM: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]