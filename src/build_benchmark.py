import os
import random
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import config
from extract_asm import load_project_functions

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ASM_DIR = os.path.join(DATA_DIR, "asm_x64")
OUTPUT_DIR = os.path.join(DATA_DIR, "bcsd_benchmark")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_PROJECTS = ['zlib', 'openssl'] 

def build_bcsd_dataset():
    print(f"Loading Test Projects: {TEST_PROJECTS} ...")
    all_test_functions = []
    
    for proj in TEST_PROJECTS:
        proj_path = os.path.join(ASM_DIR, proj)
        funcs = load_project_functions(proj_path)
        all_test_functions.extend(funcs)
        print(f"  - Loaded {proj}: {len(funcs)} functions")
        
    print(f"Total functions in pool: {len(all_test_functions)}")
    
    # functions with the same func_name considered to be the same semantic funciton
    # e.g. {"func_1": [1, 99]}
    func_name_to_indices = defaultdict(list)
    for idx, item in enumerate(all_test_functions):
        func_name_to_indices[item['func_name']].append(idx)
        
    # query use func_name
    # positive sample must more than 2
    # e.g. ["func_1", "func_55"]
    valid_query_names = [k for k, v in func_name_to_indices.items() if len(v) >= 2]
    print(f"Unique function names with >= 2 variants: {len(valid_query_names)}")
    
    if len(valid_query_names) < 100:
        print("Error: Not enough valid functions to build queries! Check dataset.")
        return

    # build BCSD
    # 1000 queries, 10000 pool
    
    NUM_QUERIES = 1000
    POOL_SIZE = 100000
    
    NUM_QUERIES = min(NUM_QUERIES, len(valid_query_names))
    
    random.seed(42) # fixed seed for reproduction
    selected_query_names = random.sample(valid_query_names, NUM_QUERIES)
    
    queries = []       # [ {asm_text, func_name, ground_truth_key}, ... ]
    pool_candidates = [] # [ {asm_text, func_name, ...}, ... ]
    
    query_indices = set()
    pool_indices = set()
    
    ground_truth = {} # e.g. { query_index: [pool_index_1, pool_index_2...] }
    
    print("Constructing Query Set and Gallery Pool...")
    
    # build query
    for q_name in selected_query_names:
        variants_indices = func_name_to_indices[q_name]
        # choose one as query
        q_idx = random.choice(variants_indices)
        query_indices.add(q_idx)
        
        queries.append(all_test_functions[q_idx])
        
        # put rest of variants(positive sample) into pool
        for v_idx in variants_indices:
            if v_idx != q_idx:
                pool_indices.add(v_idx)
                
    # fill pool, choose negatives randomly
    remaining_indices = []
    for i in range(len(all_test_functions)):
        if i not in query_indices and i not in pool_indices:
            remaining_indices.append(i)
            
    needed = POOL_SIZE - len(pool_indices)
    if needed > 0:
        fillers = random.sample(remaining_indices, min(needed, len(remaining_indices)))
        pool_indices.update(fillers)
    
    # turn pool_indices from set into list
    final_pool_indices_list = list(pool_indices)
    
    # final pool
    pool_candidates = [all_test_functions[i] for i in final_pool_indices_list]
    
    # generate ground truth
    # need to map：Query[i] -> which indices in pool are correct
    # create helper dict：func_name -> pool pndices
    name_to_pool_ids = defaultdict(list)
    for pool_idx, item in enumerate(pool_candidates):
        name_to_pool_ids[item['func_name']].append(pool_idx)
        
    for q_i, q_item in enumerate(queries):
        target_name = q_item['func_name']
        correct_pool_ids = name_to_pool_ids.get(target_name, [])
        ground_truth[str(q_i)] = correct_pool_ids # {"0": [1,99]}
        
    # save results
    print(f"Final Stat:")
    print(f"  - Queries: {len(queries)}")
    print(f"  - Pool Size: {len(pool_candidates)}")
    
    with open(os.path.join(OUTPUT_DIR, "bcsd_queries.json"), "w") as f:
        json.dump(queries, f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "bcsd_pool.json"), "w") as f:
        json.dump(pool_candidates, f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "bcsd_ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Saved benchmark to {OUTPUT_DIR}")


if __name__ == "__main__":
    build_bcsd_dataset()