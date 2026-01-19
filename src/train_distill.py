import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import SmallBERT
from dataset_distill import DistillationDataset
from utils_eval import evaluate_model
import config

EPOCHS = 20          
BATCH_SIZE = 128   
LR = 5e-5
TEMPERATURE = 2.0     
ALPHA = 1.0           # 1.0 = pure KD, 0.5 = KD+Triplet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_DIR = os.path.join(PROJECT_ROOT, "data", "outputs", "student", "128")
TEACHER_DIR = os.path.join(PROJECT_ROOT, "data", "outputs", "teacher", "256")
SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "checkpoints", "distill_kl")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "bcsd_benchmark_1")

os.makedirs(SAVE_DIR, exist_ok=True)

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_embeds, teacher_embeds):
        """
        compute Listwise KD Loss
        Args:
            student_embeds: [Batch, 256]
            teacher_embeds: [Batch, 768]
        """
        s_norm = F.normalize(student_embeds, p=2, dim=1)
        t_norm = F.normalize(teacher_embeds, p=2, dim=1)
        
        # compute similarity metrix [Batch, Batch]
        sim_stu = torch.mm(s_norm, s_norm.t())
        sim_tea = torch.mm(t_norm, t_norm.t())
        
        # temperatur + LogSoftmax / Softmax
        # Student needs log (KLDivLoss requirement)
        log_prob_stu = F.log_softmax(sim_stu / self.T, dim=1)
        prob_tea = F.softmax(sim_tea / self.T, dim=1)
        
        # compute KL divergence
        loss = (self.T ** 2) * self.kl_div(log_prob_stu, prob_tea)
        
        return loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Batch Size: {BATCH_SIZE}, Temp: {TEMPERATURE})")

    print("Initializing Distillation Dataset...")
    # automatic alignment
    train_dataset = DistillationDataset(STUDENT_DIR, TEACHER_DIR, train_projects=['nmap','openssl','unrar','zlib'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0, 
        drop_last=True      # drop last incomplete Batch
    )
    
    # initialize Student
    print("Initializing Student Model...")
    model = SmallBERT().to(device) 
    
    optimizer = AdamW(model.parameters(), lr=LR)
    kd_criterion = DistillationLoss(temperature=TEMPERATURE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps*0.1), 
        num_training_steps=total_steps
    )

    best_mrr = 0.0
    print("Start Knowledge Distillation Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            teacher_embeds = batch['teacher_embed'].to(device) # [Batch, 768]
            
            # Student forward
            student_embeds = model(input_ids, attention_mask, token_type_ids) # [Batch, 256]

            loss = kd_criterion(student_embeds, teacher_embeds)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg KD Loss: {avg_loss:.4f}")
        
        # evaluation
        print("Evaluating Student on BCSD Benchmark...")
        mrr, recall = evaluate_model(model, device, BENCHMARK_DIR)
        print(f"Student Result: MRR@10 = {mrr:.4f}, Recall@1 = {recall:.4f}")
        
        # save
        if mrr > best_mrr:
            best_mrr = mrr
            save_path = os.path.join(SAVE_DIR, "best_student_model")
            model.save_pretrained(save_path)
            print(f"New Best Student Model Saved! (MRR: {best_mrr:.4f})")

    print("Distillation Finished.")

if __name__ == "__main__":
    main()