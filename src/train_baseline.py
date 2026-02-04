import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from model import SmallBERT
from dataset_triplet import BCSDTripletDataset
from model_eval import evaluate_model
import config

EPOCHS = 20
EPOCH_SAMPLE_RATE = 15
BATCH_SIZE = 64
LR = 5e-5

SAVE_DIR = os.path.join(config.DATA_DIR, "checkpoints", "baseline")
BENCHMARK_DIR = os.path.join(config.DATA_DIR, "bcsd_benchmark_5")
os.makedirs(SAVE_DIR, exist_ok=True)

class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        # normalize vectors
        a_norm = torch.nn.functional.normalize(anchor, p=2, dim=1)
        p_norm = torch.nn.functional.normalize(pos, p=2, dim=1)
        n_norm = torch.nn.functional.normalize(neg, p=2, dim=1)
        
        # calculate cosine similarity
        sim_pos = torch.sum(a_norm * p_norm, dim=1)
        sim_neg = torch.sum(a_norm * n_norm, dim=1)
        
        # calculate loss
        losses = torch.relu(self.margin - sim_pos + sim_neg)
        return losses.mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing Dataset...")
    train_dataset = BCSDTripletDataset(
        projects=['openssl', 'clamav', 'zlib', 'nmap'],
        epoch_sample_rate=EPOCH_SAMPLE_RATE
    )

    # Iterable, len(train_loader)=N/B
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    print("Initializing SmallBERT Baseline...")
    model = SmallBERT().to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = TripletCosineLoss(margin=0.2)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_map = 0.0
    
    print("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
        
            # [batch, triple, seq_len]
            # second dim: [Anchor_1, Positive_1, Negative_1]
            b_size, n_triplet, seq_len = input_ids.shape
            
            # flatten, because bert can process 2 dims
            flat_input_ids = input_ids.view(-1, seq_len)
            flat_mask = attention_mask.view(-1, seq_len)
            flat_token_type_ids = token_type_ids.view(-1, seq_len)
            
            embeddings = model(flat_input_ids, flat_mask, flat_token_type_ids)
            
            # restore shape [Batch, 3, Hidden]
            embeddings = embeddings.view(b_size, n_triplet, -1)
            
            anchor = embeddings[:, 0, :]
            positive = embeddings[:, 1, :]
            negative = embeddings[:, 2, :]
            
            loss = criterion(anchor, positive, negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # evaluation
        eval_results = evaluate_model(model, device, BENCHMARK_DIR, mode="val")
        current_map = eval_results.get("Map@50", 0.0)
        current_pre = eval_results.get("R-Precision", 0.0)
        print(f"Result: Map@50 = {current_map:.4f}, R-Precision = {current_pre:.4f}")
        
        # save model
        if current_map > best_map:
            best_map = current_map
            save_path = os.path.join(SAVE_DIR, "baseline_model")
            model.save_pretrained(save_path)
            print(f"New Best Baseline Model Saved to {save_path}!")

    print("Training Finished.")

    print("\n=== Starting Final Evaluation on Test Set ===")
    
    best_model_path = os.path.join(SAVE_DIR, "baseline_model")
    
    try:
        best_model = SmallBERT.from_pretrained(best_model_path)
    except:
        best_model = SmallBERT()
        print("Warning: Could not use from_pretrained. Please check model loading logic.")
    
    best_model = best_model.to(device)
    test_results = evaluate_model(best_model, device, BENCHMARK_DIR, mode="test")
    
    print(f"\n>>>>>> FINAL TEST RESULTS <<<<<<")
    print(f"Baseline Model: {best_model_path}")
    for metric_name, score in test_results.items():
        print(f"{metric_name:<15}: {score:.4f}")
    print(f">>>>>>>>>>>>>><<<<<<<<<<<<<<")

if __name__ == "__main__":
    main()