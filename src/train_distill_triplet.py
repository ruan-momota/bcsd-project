import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from model import SmallBERT
from dataset_triplet import BCSDTripletDataset
from utils_eval import evaluate_model
import config

EPOCHS = 20                 
EPOCH_SAMPLE_RATE = 20
BATCH_SIZE = 64
LR = 3e-5       

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_DIR = os.path.join(DATA_DIR, "bcsd_benchmark")
DISTILLED_MODEL_PATH = os.path.join(DATA_DIR, "checkpoints", "distill", "best_student_model")
SAVE_DIR = os.path.join(DATA_DIR, "checkpoints", "dis_trip")
os.makedirs(SAVE_DIR, exist_ok=True)


class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        a_norm = torch.nn.functional.normalize(anchor, p=2, dim=1)
        p_norm = torch.nn.functional.normalize(pos, p=2, dim=1)
        n_norm = torch.nn.functional.normalize(neg, p=2, dim=1)
        
        sim_pos = torch.sum(a_norm * p_norm, dim=1)
        sim_neg = torch.sum(a_norm * n_norm, dim=1)
        
        losses = torch.relu(self.margin - sim_pos + sim_neg)
        return losses.mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing Triplet Dataset for Fine-tuning...")
    train_dataset = BCSDTripletDataset(
        projects=['openssl', 'clamav', 'zlib', 'nmap'],
        epoch_sample_rate=EPOCH_SAMPLE_RATE
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Loading Distilled Student Model from: {DISTILLED_MODEL_PATH}")
    model = SmallBERT.from_pretrained(DISTILLED_MODEL_PATH)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = TripletCosineLoss(margin=0.2)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps*0.1), 
        num_training_steps=total_steps
    )

    print("Evaluating Initial Performance (Distilled Only)...")
    init_mrr, _ = evaluate_model(model, device, BENCHMARK_DIR, mode="test")
    print(f"Initial Distilled MRR: {init_mrr:.4f}")
    
    best_mrr = init_mrr
    
    # start fine-tuning
    print("Start Fine-tuning with Triplet Loss...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
        
            # [Batch, 3, Seq] -> Flatten -> BERT -> [Batch, 3, Hidden]
            b_size, n_triplet, seq_len = input_ids.shape
            
            flat_input_ids = input_ids.view(-1, seq_len)
            flat_mask = attention_mask.view(-1, seq_len)
            flat_token_type_ids = token_type_ids.view(-1, seq_len)
            
            embeddings = model(flat_input_ids, flat_mask, flat_token_type_ids)
            embeddings = embeddings.view(b_size, n_triplet, -1)
            
            anchor, positive, negative = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
            
            loss = criterion(anchor, positive, negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # Evaluation
        print("Evaluating...")
        mrr, recall = evaluate_model(model, device, BENCHMARK_DIR, mode="val")
        print(f"Result: MRR@10 = {mrr:.4f}, Recall@1 = {recall:.4f}")
        
        if mrr > best_mrr:
            best_mrr = mrr
            save_path = os.path.join(SAVE_DIR, "best_model")
            model.save_pretrained(save_path)
            print(f"New Best Model (Distill+Triplet) Saved! MRR: {best_mrr:.4f}")

    print("Fine-tuning Finished.")

    print("\n=== Starting Final Evaluation on Test Set ===")
    
    best_model_path = os.path.join(SAVE_DIR, "best_model")
    
    try:
        best_model = SmallBERT.from_pretrained(best_model_path)
    except:
        best_model = SmallBERT()
        print("Warning: Could not use from_pretrained. Please check model loading logic.")
    
    best_model = best_model.to(device)
    
    test_mrr, test_recall = evaluate_model(best_model, device, BENCHMARK_DIR, mode="test")
    
    print(f"\n>>>>>> FINAL TEST RESULTS <<<<<<")
    print(f"Model: {best_model_path}")
    print(f"MRR@10   : {test_mrr:.4f}")
    print(f"Recall@1 : {test_recall:.4f}")
    print(f">>>>>>>>>>>>>><<<<<<<<<<<<<<")

if __name__ == "__main__":
    main()