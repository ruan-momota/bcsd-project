import os
import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_DIR = os.path.join(DATA_DIR, "bcsd_benchmark", "test")
TEACHER_OUTPUT_ROOT = os.path.join(DATA_DIR, "outputs", "teacher", "256")
TARGET_PROJECTS = ['curl'] 

def compute_mrr_recall(similarity_matrix, ground_truth, k=10):

    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
        
    mrr_sum = 0.0
    recall_hits = 0
    num_queries = len(ground_truth)
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
                rank = i + 1 
                break
        
        if rank != -1:
            mrr_sum += 1.0 / rank
            
    mrr_score = mrr_sum / num_queries
    recall_score = recall_hits / num_queries
    
    return mrr_score, recall_score

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
        print("Benchmark directory not found. Please run build_benchmark_1.py first.")
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
    mrr, recall = compute_mrr_recall(sim_matrix, gt)

    print("\n" + "="*40)
    print(f"Teacher Precomputed Benchmark Result:")
    print(f"MRR    : {mrr:.4f}")
    print(f"Recall : {recall:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()