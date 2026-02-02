import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import random

from dataset_mlm import BCSDMLMDataset
from utils_eval import evaluate_model
from model import SmallBERT
import config

EPOCHS = 20
BATCH_SIZE = 64
LR = 5e-5
MASK_PROB = 0.15
PAD_TOKEN_ID = 1 
MASK_TOKEN_ID = 5
SAVE_DIR = os.path.join(config.DATA_DIR, "checkpoints", "mlm")
BENCHMARK_DIR = os.path.join(config.DATA_DIR, "bcsd_benchmark")
os.makedirs(SAVE_DIR, exist_ok=True)

def collate_mlm(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    
    labels = input_ids.clone()
    
    # build masked matrix
    probability_matrix = torch.full(labels.shape, MASK_PROB)
    probability_matrix.masked_fill_(labels == PAD_TOKEN_ID, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 80% [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = MASK_TOKEN_ID

    # 10% random Token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(33555, input_ids.shape, dtype=torch.long).to(input_ids.device)
    input_ids[indices_random] = random_words[indices_random]

    labels[~masked_indices] = -100 
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = BCSDMLMDataset(projects=['openssl', 'clamav', 'zlib', 'nmap'])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=0, collate_fn=collate_mlm)
    
    print("Initializing BertForMaskedLM...")
    config = BertConfig(
        vocab_size=33555,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=256,
        type_vocab_size=256,
        pad_token_id=PAD_TOKEN_ID
    )
    model = BertForMaskedLM(config)
    model.to(device) # type: ignore
    
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    
    best_mrr = 0.0

    print("Start MLM Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Done. Avg MLM Loss: {avg_loss:.4f}")
        
        # === Evaluation ===
        eval_model = SmallBERT().to(device)
        eval_model.bert = model.bert # share encoder weight
        
        mrr, recall = evaluate_model(eval_model, device, BENCHMARK_DIR, mode="val")
        print(f"Val Retrieval: MRR={mrr:.4f}, Recall={recall:.4f}")
        
        # Save Best Model
        if mrr > best_mrr:
            best_mrr = mrr
            save_path = os.path.join(SAVE_DIR, "best_mlm_model")
            model.bert.save_pretrained(save_path)
            print(f"Saved best encoder to {save_path}")

    print("MLM Training Finished.")
    
    print("\n=== Starting Final Evaluation on Test Set ===")
    
    best_model_path = os.path.join(SAVE_DIR, "best_mlm_model")
    
    try:
        best_model = SmallBERT.from_pretrained(best_model_path)
        best_model = best_model.to(device)
        print(f"Successfully loaded best model from {best_model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_mrr, test_recall = evaluate_model(best_model, device, BENCHMARK_DIR, mode="test")
    
    print(f"\n>>>>>> FINAL TEST RESULTS (MLM Zero-shot) <<<<<<")
    print(f"Model Source: {best_model_path}")
    print(f"MRR@10      : {test_mrr:.4f}")
    print(f"Recall@1    : {test_recall:.4f}")
    print(f">>>>>>>>>>>>>><<<<<<<<<<<<<<")

if __name__ == "__main__":
    main()