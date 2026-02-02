import os
import random
import json
import torch
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "outputs", "student", "256")
BLOCKLIST_FILE = os.path.join(DATA_DIR, "outputs", "blocklist256.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "bcsd_benchmark")

TRAIN_PROJECTS = ['openssl', 'clamav', 'zlib']

VAL_PROJECTS = ['unrar']
VAL_POOL_SIZE = 10000
VAL_NUM_QUERIES = 1000

TEST_PROJECTS = ['curl']
TEST_POOL_SIZE = 10000
TEST_NUM_QUERIES = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_preprocessed_project(proj_name, blocklist=None):
    proj_dir = os.path.join(INPUT_DIR, proj_name)
    if not os.path.exists(proj_dir):
        print(f"[Error] Directory {proj_dir} does not exist. Skipping.")
        return [], 0

    all_items = []
    pt_files = [f for f in os.listdir(proj_dir) if f.endswith(".pt")]
    skipped_count = 0
    
    for pt_file in tqdm(pt_files, desc=f"Loading {proj_name}", leave=False):
        full_path = os.path.join(proj_dir, pt_file)
        try:
            data = torch.load(full_path)

            if blocklist:
                clean_data = []
                for item in data:
                    unique_key = f"{item['proj_name']}|{item['file_name']}|{item['func_name']}"
                    if unique_key in blocklist:
                        skipped_count += 1
                        continue
                    clean_data.append(item)
                all_items.extend(clean_data)
            else:
                all_items.extend(data)

        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            
    return all_items, skipped_count

def load_blocklist():
    if os.path.exists(BLOCKLIST_FILE):
        with open(BLOCKLIST_FILE, 'r', encoding='utf-8') as f:
            blocklist = set(json.load(f))
        return blocklist
    else:
        print(f"[Warning] Blocklist not found. Building without filtering.")
        return set()

def build_training_set(blocklist):
    print("\n=== Building Training Set ===")
    save_dir = os.path.join(OUTPUT_DIR, "train")
    os.makedirs(save_dir, exist_ok=True)
    
    all_train_funcs = []
    total_skipped = 0
    
    for proj in TRAIN_PROJECTS:
        funcs, skipped = load_preprocessed_project(proj, blocklist)
        all_train_funcs.extend(funcs)
        total_skipped += skipped
        print(f"  - Loaded {proj}: {len(funcs)} functions (Skipped: {skipped})")
        
    print(f"Total Training Functions: {len(all_train_funcs)}")
    
    save_path = os.path.join(save_dir, "train_all.pt")
    torch.save(all_train_funcs, save_path)
    print(f"Training set saved to {save_path}")

def build_retrieval_set(projects, pool_size, num_queries, mode, blocklist):
    print(f"\n=== Building {mode.upper()} Set ({projects}) ===")
    save_dir = os.path.join(OUTPUT_DIR, mode)
    os.makedirs(save_dir, exist_ok=True)
    all_funcs = []
    total_skipped = 0
    
    for proj in projects:
        funcs, skipped = load_preprocessed_project(proj, blocklist)
        all_funcs.extend(funcs)
        total_skipped += skipped
    
    print(f"  - Total available functions: {len(all_funcs)}")
    
    if len(all_funcs) < pool_size:
        print(f"[Warning] Not enough functions for requested Pool Size {pool_size}. Using max available: {len(all_funcs)}")
        pool_size = len(all_funcs)

    # 1. Group by Function Name
    func_name_to_indices = defaultdict(list)
    for idx, item in enumerate(all_funcs):
        func_name_to_indices[item['func_name']].append(idx)
        
    # 2. Filter valid queries (Must have >= 2 compiled versions)
    valid_query_names = [k for k, v in func_name_to_indices.items() if len(v) >= 2]
    print(f"  - Unique function names with >= 2 variants: {len(valid_query_names)}")
    
    if len(valid_query_names) < 100:
        print(f"[Error] Not enough valid query candidates in {projects}!")
        return

    # 3. Select Queries
    actual_num_queries = min(num_queries, len(valid_query_names))
    random.seed(42) # Fixed seed for reproducibility
    
    selected_query_names = random.sample(valid_query_names, actual_num_queries)
    
    queries = []       
    query_indices = set()
    pool_indices = set() # Set of indices in all_funcs
    
    print("  - Constructing Query Set and Gallery Pool...")
    
    for q_name in selected_query_names:
        variants_indices = func_name_to_indices[q_name]
        
        # Strategy: Pick 1 as Query, put ALL others into Pool
        # This ensures the answer exists in the pool
        q_idx = random.choice(variants_indices)
        query_indices.add(q_idx)
        queries.append(all_funcs[q_idx])
        
        for v_idx in variants_indices:
            if v_idx != q_idx:
                pool_indices.add(v_idx)
                
    # 4. Fill the rest of the Pool with distractors
    # Candidates are those NOT used as queries and NOT already in pool
    remaining_indices = []
    for i in range(len(all_funcs)):
        if i not in query_indices and i not in pool_indices:
            remaining_indices.append(i)
            
    needed = pool_size - len(pool_indices)
    if needed > 0:
        if needed > len(remaining_indices):
            print(f"  - [Note] Pool filled with all remaining functions (Size: {len(pool_indices) + len(remaining_indices)})")
            pool_indices.update(remaining_indices)
        else:
            fillers = random.sample(remaining_indices, needed)
            pool_indices.update(fillers)
    else:
        print(f"  - [Note] Pool already larger than requested size due to variants ({len(pool_indices)})")
    
    # 5. Extract Pool Items
    final_pool_indices_list = list(pool_indices)
    pool_candidates = [all_funcs[i] for i in final_pool_indices_list]
    
    # 6. Generate Ground Truth
    # Format: {"0": [pool_idx_1, pool_idx_2], "1": [...]}
    ground_truth = {} 
    
    # Create a lookup for pool: func_name -> list of pool indices
    name_to_pool_ids = defaultdict(list)
    for pool_idx, item in enumerate(pool_candidates):
        name_to_pool_ids[item['func_name']].append(pool_idx)
        
    for q_i, q_item in enumerate(queries):
        target_name = q_item['func_name']
        correct_pool_ids = name_to_pool_ids.get(target_name, [])
        ground_truth[str(q_i)] = correct_pool_ids 
        
    # Save Results
    torch.save(queries, os.path.join(save_dir, f"{mode}_queries.pt"))
    torch.save(pool_candidates, os.path.join(save_dir, f"{mode}_pool.pt"))
    with open(os.path.join(save_dir, f"{mode}_ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
        
    print(f"  - Saved {mode} set: {len(queries)} Queries, {len(pool_candidates)} Pool size.")

if __name__ == "__main__":
    # Load blocklist once
    blk = load_blocklist()
    
    # 1. Build Training Set
    # build_training_set(blk)
    
    # 2. Build Validation Set (Unrar)
    build_retrieval_set(VAL_PROJECTS, VAL_POOL_SIZE, VAL_NUM_QUERIES, "val", blk)
    
    # 3. Build Test Set (Curl)
    build_retrieval_set(TEST_PROJECTS, TEST_POOL_SIZE, TEST_NUM_QUERIES, "test", blk)
    
    print("\nAll datasets built successfully!")