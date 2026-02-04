import torch
import numpy as np
from tqdm import tqdm
import json
import os

BATCH_SIZE = 128

def compute_metrics(similarity_matrix, ground_truth, k_list=[1,10,20,50]):
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
        
    metrics = {
        f"Map@{k}": 0.0 for k in k_list
    }
    metrics.update({f"Ndcg@{k}": 0.0 for k in k_list})
    metrics.update({f"Precision@{k}": 0.0 for k in k_list})
    metrics.update({f"Recall@{k}": 0.0 for k in k_list})
    metrics["R-Precision"] = 0.0
    
    num_queries = len(ground_truth)
    
    max_k = max(k_list)
    safe_depth = max(max_k, 200)
    sorted_indices = np.argsort(-similarity_matrix, axis=1)[:, :safe_depth]
    
    for q_idx_int, candidates in enumerate(sorted_indices):
        q_idx_str = str(q_idx_int)
        if q_idx_str not in ground_truth:
            continue
            
        correct_pool_ids = set(ground_truth[q_idx_str])
        total_rel = len(correct_pool_ids)  # R
        
        if total_rel == 0:
            continue

        # --- R-Precision ---
        r_cutoff = min(total_rel, safe_depth)
        r_candidates = candidates[:r_cutoff]
        r_hits_count = np.sum([1 if c in correct_pool_ids else 0 for c in r_candidates])
        metrics["R-Precision"] += r_hits_count / total_rel

        # normal metric
        hits = np.array([1 if c in correct_pool_ids else 0 for c in candidates[:max_k]])
        
        for k in k_list:
            hits_k = hits[:k]
            num_hits = np.sum(hits_k)
            
            # --- Precision@K ---
            metrics[f"Precision@{k}"] += num_hits / k
            
            # --- Recall@K ---
            metrics[f"Recall@{k}"] += num_hits / total_rel
            
            # --- MAP@K ---
            if num_hits > 0:
                precisions = np.cumsum(hits_k) / np.arange(1, k + 1)
                ap = np.sum(precisions * hits_k) / min(k, total_rel)
                metrics[f"Map@{k}"] += ap
            
            # --- NDCG@K ---
            if num_hits > 0:
                dcg = np.sum((2**hits_k - 1) / np.log2(np.arange(2, k + 2)))
                ideal_hits = np.zeros(k)
                ideal_hits[:min(k, total_rel)] = 1
                idcg = np.sum((2**ideal_hits - 1) / np.log2(np.arange(2, k + 2)))
                metrics[f"Ndcg@{k}"] += dcg / idcg

    # mean
    for key in metrics:
        metrics[key] /= num_queries
        
    return metrics

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
        return {}

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

    # Encoding
    query_tensor = get_embeddings(queries) 
    pool_tensor = get_embeddings(pool)     
    
    # Normalize
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
    pool_norm = torch.nn.functional.normalize(pool_tensor, p=2, dim=1)
    
    # Similarity Matrix
    sim_matrix = torch.mm(query_norm, pool_norm.t())

    results = compute_metrics(sim_matrix, gt, k_list=[1, 10, 20, 50])

    # print("Evaluation Results:")
    # for metric, score in results.items():
    #     print(f"{metric:<15}: {score:.4f}")
    
    model.train()

    return results