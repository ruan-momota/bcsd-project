import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class DistillationDataset(Dataset):
    def __init__(self, student_dir, teacher_dir, train_projects=None):
        self.inputs = []
        
        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_teacher_embeds = []

        if not os.path.exists(student_dir):
            raise ValueError(f"Student directory does not exist: {student_dir}")
            
        available_projects = [d for d in os.listdir(student_dir) if os.path.isdir(os.path.join(student_dir, d))]
        
        if train_projects is None:
            projects_to_process = available_projects
        else:
            projects_to_process = [p for p in available_projects if p in train_projects]

        print(f"Loading data from projects: {projects_to_process}")

        total_files = 0
        total_samples = 0

        for project in tqdm(projects_to_process, desc="Loading Projects"):
            s_proj_path = os.path.join(student_dir, project)
            t_proj_path = os.path.join(teacher_dir, project)

            if not os.path.exists(t_proj_path):
                print(f"Warning: Teacher folder for project '{project}' not found. Skipping.")
                continue

            pt_files = [f for f in os.listdir(s_proj_path) if f.endswith(".pt")]

            for pt_file in pt_files:
                s_file_path = os.path.join(s_proj_path, pt_file)
                t_file_path = os.path.join(t_proj_path, pt_file)

                try:
                    s_data = torch.load(s_file_path)
                    t_data = torch.load(t_file_path)

                except Exception as e:
                    print(f"Error loading {pt_file}: {e}")
                    continue

                if len(s_data) != len(t_data):
                    print(f"Mismatch length in {project}/{pt_file}: Student {len(s_data)} vs Teacher {len(t_data)}")
                    continue
                
                for i in range(len(s_data)):
                    s_item = s_data[i]
                    t_item = t_data[i]
                    
                    # double check
                    if s_item['func_name'] != t_item['func_name']:
                        print("diff func!!!")
                        continue

                    # Student Inputs
                    s_inputs = s_item['student_input']
                    all_input_ids.append(s_inputs['input_ids'])
                    all_attention_mask.append(s_inputs['attention_mask'])
                    all_token_type_ids.append(s_inputs['token_type_ids'])
                    
                    # Teacher Target
                    all_teacher_embeds.append(t_item['teacher_embed'])
                    
                total_files += 1
                total_samples += len(s_data)

        print(f"Data Loading Complete.")
        print(f"Processed Files: {total_files}")
        print(f"Total Samples: {total_samples}")

        if total_samples == 0:
            raise ValueError("No valid training data found. Check your paths.")

        self.input_ids = torch.stack(all_input_ids)
        self.attention_mask = torch.stack(all_attention_mask)
        self.token_type_ids = torch.stack(all_token_type_ids)
        self.teacher_vecs = torch.stack(all_teacher_embeds)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'teacher_embed': self.teacher_vecs[idx]
        }