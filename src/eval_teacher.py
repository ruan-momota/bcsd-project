import os
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import config

BENCHMARK_DIR = os.path.join(config.DATA_DIR, "bcsd_benchmark_5", "test")
TEACHER_OUTPUT_ROOT = os.path.join(config.DATA_DIR, "outputs", "teacher", "256_5")
TARGET_PROJECTS = ['curl'] 

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

        # --- R-Precision 计算 ---
        r_cutoff = min(total_rel, safe_depth)
        r_candidates = candidates[:r_cutoff]
        r_hits_count = np.sum([1 if c in correct_pool_ids else 0 for c in r_candidates])
        metrics["R-Precision"] += r_hits_count / total_rel

        hits = np.array([1 if c in correct_pool_ids else 0 for c in candidates[:max_k]])
        
        for k in k_list:
            hits_k = hits[:k]
            num_hits = np.sum(hits_k)
            
            # Precision@K
            metrics[f"Precision@{k}"] += num_hits / k
            
            # Recall@K
            metrics[f"Recall@{k}"] += num_hits / total_rel
            
            # MAP@K
            if num_hits > 0:
                precisions = np.cumsum(hits_k) / np.arange(1, k + 1)
                ap = np.sum(precisions * hits_k) / min(k, total_rel)
                metrics[f"Map@{k}"] += ap
            
            # NDCG@K
            if num_hits > 0:
                dcg = np.sum((2**hits_k - 1) / np.log2(np.arange(2, k + 2)))
                ideal_hits = np.zeros(k)
                ideal_hits[:min(k, total_rel)] = 1
                idcg = np.sum((2**ideal_hits - 1) / np.log2(np.arange(2, k + 2)))
                metrics[f"Ndcg@{k}"] += dcg / idcg

    for key in metrics:
        metrics[key] /= num_queries
        
    return metrics

def load_teacher_embeddings_map():
    print(f"Scanning Teacher Embeddings in: {TEACHER_OUTPUT_ROOT} ...")
    
    embedding_map = {}
    file_count = 0
    
    for root, dirs, files in os.walk(TEACHER_OUTPUT_ROOT):
        path_parts = root.split(os.sep)
        if not any(proj in path_parts for proj in TARGET_PROJECTS):
            continue

        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.join(root, file)
                try:
                    data = torch.load(full_path, map_location='cpu')
                    if isinstance(data, list):
                        for item in data:
                            if item['proj_name'] in TARGET_PROJECTS:
                                key = (item['proj_name'], item['file_name'], item['func_name'])
                                embedding_map[key] = item['teacher_embed']
                    
                    file_count += 1
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")

    print(f"Loaded {len(embedding_map)} embeddings from {file_count} files.")
    return embedding_map

def align_data_with_embeddings(data_list, embedding_map, name="Data"):
    vecs = []
    missing_count = 0
    
    embed_dim = 768
    if embedding_map:
        first_key = next(iter(embedding_map))
        embed_dim = embedding_map[first_key].shape[0]

    for item in tqdm(data_list, desc=f"Aligning {name}"):
        key = (item['proj_name'], item['file_name'], item['func_name'])
        
        if key in embedding_map:
            vecs.append(embedding_map[key])
        else:
            missing_count += 1
            vecs.append(torch.zeros(embed_dim))

    if missing_count > 0:
        print(f"Warning: {missing_count} items in {name} were missing Teacher embeddings (Filled with Zeros).")
        
    if len(vecs) == 0:
        return torch.tensor([])
        
    return torch.stack(vecs)

def main():
    print(f"Checking Benchmark Directory: {BENCHMARK_DIR}")
    if not os.path.exists(BENCHMARK_DIR):
        print("Benchmark directory not found. Please run build_benchmark.py first.")
        return

    print("Loading Benchmark Metadata...")
    queries = torch.load(os.path.join(BENCHMARK_DIR, "test_queries.pt"))
    pool = torch.load(os.path.join(BENCHMARK_DIR, "test_pool.pt"))
    with open(os.path.join(BENCHMARK_DIR, "test_ground_truth.json"), 'r') as f:
        gt = json.load(f)

    print(f"Queries: {len(queries)}, Pool: {len(pool)}")

    emb_map = load_teacher_embeddings_map()

    if not emb_map:
        print("No embeddings loaded. Check TEACHER_OUTPUT_ROOT path.")
        return

    query_tensor = align_data_with_embeddings(queries, emb_map, "Queries")
    pool_tensor = align_data_with_embeddings(pool, emb_map, "Pool")

    # release map
    del emb_map

    print("\nCalculating Similarity Matrix...")
    if torch.cuda.is_available():
        query_tensor = query_tensor.cuda()
        pool_tensor = pool_tensor.cuda()
        print("Moved tensors to GPU.")

    # L2 Normalization
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
    pool_norm = torch.nn.functional.normalize(pool_tensor, p=2, dim=1)

    # Cosine Similarity: [Q, D] * [D, P] -> [Q, P]
    sim_matrix = torch.mm(query_norm, pool_norm.t())

    print("Computing Metrics...")
    results = compute_metrics(sim_matrix, gt, k_list=[1, 10, 20, 50])

    print("\n" + "="*40)
    print(f"Teacher Precomputed Benchmark Result:")
    for metric, score in results.items():
        print(f"{metric:<15}: {score:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()