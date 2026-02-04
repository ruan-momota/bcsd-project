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
from model_eval import evaluate_model
import config

EPOCHS = 20          
BATCH_SIZE = 64   
LR = 5e-5
TEMPERATURE = 2.0     
ALPHA = 1.0           # 1.0 = pure KD, 0.5 = KD+Triplet

STUDENT_DIR = os.path.join(config.DATA_DIR, "outputs", "student", "256_5")
TEACHER_DIR = os.path.join(config.DATA_DIR, "outputs", "teacher", "256_5")
SAVE_DIR = os.path.join(config.DATA_DIR, "checkpoints", "disti")
BENCHMARK_DIR = os.path.join(config.DATA_DIR, "bcsd_benchmark_5")
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
    train_dataset = DistillationDataset(STUDENT_DIR, TEACHER_DIR, train_projects=['openssl', 'clamav', 'zlib', 'nmap'])
    
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

    best_map = 0.0
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
        eval_results = evaluate_model(model, device, BENCHMARK_DIR, mode="val")
        current_map = eval_results.get("Map@50", 0.0)
        current_pre = eval_results.get("R-Precision", 0.0)
        print(f"Result: Map@50 = {current_map:.4f}, R-Precision = {current_pre:.4f}")
        
        # save
        if current_map > best_map:
            best_map = current_map
            save_path = os.path.join(SAVE_DIR, "student_model")
            model.save_pretrained(save_path)
            print(f"New Best Distillation Model Saved to {save_path}!")

    print("Distillation Finished.")

    print("\n=== Starting Final Evaluation on Test Set ===")
    
    best_model_path = os.path.join(SAVE_DIR, "student_model")
    
    try:
        best_model = SmallBERT.from_pretrained(best_model_path)
    except:
        best_model = SmallBERT()
        print("Warning: Could not use from_pretrained. Please check model loading logic.")
    
    best_model = best_model.to(device)
    
    test_results = evaluate_model(best_model, device, BENCHMARK_DIR, mode="test")
    
    print(f"\n>>>>>> FINAL TEST RESULTS <<<<<<")
    print(f"Student Model: {best_model_path}")
    for metric_name, score in test_results.items():
        print(f"{metric_name:<15}: {score:.4f}")
    print(f">>>>>>>>>>>>>><<<<<<<<<<<<<<")


if __name__ == "__main__":
    main()