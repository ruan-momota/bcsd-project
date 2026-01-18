import os
import random
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TOKENIZER_LEN = 128 

INPUT_DIR = os.path.join(DATA_DIR, "outputs", "student", str(TOKENIZER_LEN))
OUTPUT_DIR = os.path.join(DATA_DIR, "bcsd_benchmark")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_PROJECTS = ['z3', 'openssl'] 

def load_preprocessed_project(proj_name):
    proj_dir = os.path.join(INPUT_DIR, proj_name)
    if not os.path.exists(proj_dir):
        print(f"Warning: Directory {proj_dir} does not exist. Skipping.")
        return []

    all_items = []
    pt_files = [f for f in os.listdir(proj_dir) if f.endswith(".pt")]
    
    # leave=False: only current processbar will be on the screen
    for pt_file in tqdm(pt_files, desc=f"Loading {proj_name}", leave=False):
        full_path = os.path.join(proj_dir, pt_file)
        try:
            data = torch.load(full_path)
            all_items.extend(data)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            
    return all_items

def build_bcsd_dataset():
    print(f"Loading Test Projects from pre-tokenized inputs: {TEST_PROJECTS} ...")
    all_test_functions = []
    
    for proj in TEST_PROJECTS:
        funcs = load_preprocessed_project(proj)
        all_test_functions.extend(funcs)
        print(f"  - Loaded {proj}: {len(funcs)} functions")
        
    print(f"Total functions in pool: {len(all_test_functions)}")

    func_name_to_indices = defaultdict(list)
    for idx, item in enumerate(all_test_functions):
        func_name_to_indices[item['func_name']].append(idx)
        
    valid_query_names = [k for k, v in func_name_to_indices.items() if len(v) >= 2]
    print(f"Unique function names with >= 2 variants: {len(valid_query_names)}")
    
    if len(valid_query_names) < 100:
        print("Error: Not enough valid functions to build queries! Check dataset.")
        return

    NUM_QUERIES = 1000
    POOL_SIZE = 100000
    
    NUM_QUERIES = min(NUM_QUERIES, len(valid_query_names))
    
    random.seed(42) 
    # [func_name]
    selected_query_names = random.sample(valid_query_names, NUM_QUERIES)
    
    queries = []       
    
    query_indices = set()
    pool_indices = set()
    
    ground_truth = {} 
    
    print("Constructing Query Set and Gallery Pool...")
    
    # build query
    for q_name in selected_query_names:
        # all variants of one func
        variants_indices = func_name_to_indices[q_name]
        q_idx = random.choice(variants_indices)
        query_indices.add(q_idx)
        
        queries.append(all_test_functions[q_idx])
        
        # put rest variants into pool
        for v_idx in variants_indices:
            if v_idx != q_idx:
                pool_indices.add(v_idx)
                
    remaining_indices = []
    for i in range(len(all_test_functions)):
        if i not in query_indices and i not in pool_indices:
            remaining_indices.append(i)
            
    needed = POOL_SIZE - len(pool_indices)
    if needed > 0:
        if needed > len(remaining_indices):
            pool_indices.update(remaining_indices)
        else:
            fillers = random.sample(remaining_indices, needed)
            pool_indices.update(fillers)
    
    final_pool_indices_list = list(pool_indices)
    
    pool_candidates = [all_test_functions[i] for i in final_pool_indices_list]
    
    # generate ground truth
    name_to_pool_ids = defaultdict(list)
    for pool_idx, item in enumerate(pool_candidates):
        name_to_pool_ids[item['func_name']].append(pool_idx)
        
    for q_i, q_item in enumerate(queries):
        target_name = q_item['func_name']
        correct_pool_ids = name_to_pool_ids.get(target_name, [])
        ground_truth[str(q_i)] = correct_pool_ids 
        
    # save results
    print(f"Final Stat:")
    print(f"  - Queries: {len(queries)}")
    print(f"  - Pool Size: {len(pool_candidates)}")
    
    torch.save(queries, os.path.join(OUTPUT_DIR, "bcsd_queries.pt"))
    torch.save(pool_candidates, os.path.join(OUTPUT_DIR, "bcsd_pool.pt"))
    with open(os.path.join(OUTPUT_DIR, "bcsd_ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Saved benchmark to {OUTPUT_DIR}")
    print("  - bcsd_queries.pt")
    print("  - bcsd_pool.pt")
    print("  - bcsd_ground_truth.json")


if __name__ == "__main__":
    build_bcsd_dataset()