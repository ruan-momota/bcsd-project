import torch
import numpy as np
from tqdm import tqdm
import json
import os

BATCH_SIZE = 128

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
def evaluate_model(model, device, benchmark_dir, mode="val"):
    data_dir = os.path.join(benchmark_dir, mode)
    queries_path = os.path.join(data_dir, f"{mode}_queries.pt")
    pool_path = os.path.join(data_dir, f"{mode}_pool.pt")
    gt_path = os.path.join(data_dir, f"{mode}_ground_truth.json")

    try:
        queries = torch.load(queries_path)
        pool = torch.load(pool_path)
        
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            
    except FileNotFoundError as e:
        print(f"Benchmark files not found in {data_dir}: {e}")
        return 0.0, 0.0

    print(f"Evaluating on {mode.upper()} set (Queries: {len(queries)}, Pool: {len(pool)})...")

    model.eval()
    
    def get_embeddings(data_list, batch_size=BATCH_SIZE):
        vecs = []
        for i in range(0, len(data_list), batch_size):
            batch_items = data_list[i : i + batch_size]
            
            input_ids = torch.stack([item['student_input']['input_ids'] for item in batch_items]).to(device)
            attention_mask = torch.stack([item['student_input']['attention_mask'] for item in batch_items]).to(device)
            token_type_ids = torch.stack([item['student_input']['token_type_ids'] for item in batch_items]).to(device)

            batch_vecs = model(input_ids, attention_mask, token_type_ids)
            vecs.append(batch_vecs.cpu())
            
        if len(vecs) > 0:
            return torch.cat(vecs, dim=0)
        else:
            return torch.tensor([])

    # Encoding Queries
    query_tensor = get_embeddings(queries) # [Q, Hidden_Dim]

    # Encoding Pool
    pool_tensor = get_embeddings(pool)     # [P, Hidden_Dim]
    
    # normalize vectors before calculating sim matrix (Cosine Similarity)
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
    pool_norm = torch.nn.functional.normalize(pool_tensor, p=2, dim=1)
    
    # matrix multiplication
    # [Q, H] x [H, P] = [Q, P]
    sim_matrix = torch.mm(query_norm, pool_norm.t())
    
    mrr, recall = compute_mrr_recall(sim_matrix, gt)
    
    model.train()
    return mrr, recall