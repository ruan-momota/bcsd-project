import torch
import numpy as np
from tqdm import tqdm
import json
import os

def compute_mrr_recall(similarity_matrix, ground_truth, k=10):

    # similarity_matrix: [num_queries, pool_size] (Tensor or Numpy)
    # ground_truth: dict { str(query_idx): [pool_idx1, pool_idx2...] }
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
        
    mrr_sum = 0.0
    recall_hits = 0
    num_queries = len(ground_truth)
    
    # get indices in descending order
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    for q_idx, candidates in enumerate(sorted_indices):
        q_idx_str = str(q_idx)
        if q_idx_str not in ground_truth:
            continue
            
        correct_pool_ids = set(ground_truth[q_idx_str])
        
        # --- Recall@1 ---
        if candidates[0] in correct_pool_ids:
            recall_hits += 1
            
        # --- MRR@K ---
        rank = -1
        top_k = candidates[:k]
        for i, pool_id in enumerate(top_k):
            if pool_id in correct_pool_ids:
                rank = i + 1 # ranking from 1
                break
        
        if rank != -1:
            mrr_sum += 1.0 / rank
            
    mrr_score = mrr_sum / num_queries
    recall_score = recall_hits / num_queries
    
    return mrr_score, recall_score

@torch.no_grad()
def evaluate_model(model, tokenizer, device, benchmark_dir):
    # load Benchmark data
    try:
        with open(os.path.join(benchmark_dir, "bcsd_queries.json"), 'r') as f:
            queries = json.load(f)
        with open(os.path.join(benchmark_dir, "bcsd_pool.json"), 'r') as f:
            pool = json.load(f)
        with open(os.path.join(benchmark_dir, "bcsd_ground_truth.json"), 'r') as f:
            gt = json.load(f)
    except FileNotFoundError:
        print("Benchmark files not found.")
        return 0.0, 0.0

    model.eval()
    
    # encoding Queries
    query_vecs = []
    # print("Encoding Queries...")
    eval_batch = 32
    
    for i in range(0, len(queries), eval_batch):
        batch = queries[i : i+eval_batch]
        texts = [item['asm_text'] for item in batch]
        tokenizer.model_max_length = 512
        inputs = tokenizer(texts, 
                           padding='max_length', 
                           max_length=512, 
                           return_tensors='pt').to(device)
        vecs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        query_vecs.append(vecs.cpu())
    query_tensor = torch.cat(query_vecs, dim=0) # [Q, 256]

    # encoding Pool
    # print("Encoding Pool...")
    pool_vecs = []
    for i in range(0, len(pool), eval_batch):
        batch = pool[i : i+eval_batch]
        texts = [item['asm_text'] for item in batch]
        tokenizer.model_max_length = 512
        inputs = tokenizer(texts, 
                           padding='max_length', 
                           max_length=512, 
                           return_tensors='pt').to(device)
        vecs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        pool_vecs.append(vecs.cpu())
    pool_tensor = torch.cat(pool_vecs, dim=0) # [P, 256]
    
    # normalize vectors before calculating sim matrix
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
    pool_norm = torch.nn.functional.normalize(pool_tensor, p=2, dim=1)
    
    # matrix multiplication
    sim_matrix = torch.mm(query_norm, pool_norm.t())
    
    mrr, recall = compute_mrr_recall(sim_matrix, gt)
    
    model.train() # switch to training mode
    return mrr, recall