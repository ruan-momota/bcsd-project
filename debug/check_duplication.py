import os
import torch
from tqdm import tqdm
import hashlib
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "outputs", "teacher", "256")

EXCLUDE_PROJECTS = []

def get_embedding_fingerprint(tensor):
    tensor_bytes = tensor.cpu().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory {INPUT_DIR} does not exist.")
        return
    
    fingerprint_to_name = {}
    
    stats = {
        "total_scanned": 0,
        "exact_dups_ignored": 0,
        "renamed_dups_found": 0, 
    }
    
    dirty_proj_stats = defaultdict(int)
    
    conflict_examples = [] 

    target_files = []
    for project_name in os.listdir(INPUT_DIR):
        project_path = os.path.join(INPUT_DIR, project_name)
        
        if not os.path.isdir(project_path):
            continue
            
        if project_name in EXCLUDE_PROJECTS:
            continue
        
        for f in os.listdir(project_path):
            if f.endswith(".pt"):
                target_files.append(os.path.join(project_path, f))

    print(f"Found {len(target_files)} .pt files. Starting analysis...")

    for pt_file in tqdm(target_files, desc="Analyzing", unit="file"):
        try:
            data = torch.load(pt_file, map_location='cpu')
            
            if not data:
                continue
                
            for item in data:
                stats["total_scanned"] += 1
                
                current_proj = item.get('proj_name', 'unknown')
                current_func_name = item.get('func_name', 'unknown')
                embedding = item['teacher_embed']
                
                fp = get_embedding_fingerprint(embedding)
                
                if fp in fingerprint_to_name:
                    original_name = fingerprint_to_name[fp]
                    
                    if current_func_name != original_name:
                        stats["renamed_dups_found"] += 1
                        dirty_proj_stats[current_proj] += 1
                        
                        if len(conflict_examples) < 5:
                            conflict_examples.append(
                                f"Hash: {fp[:8]}... | Original: '{original_name}' vs Current: '{current_func_name}' (in {current_proj})"
                            )
                    else:
                        stats["exact_dups_ignored"] += 1
                        
                else:
                    fingerprint_to_name[fp] = current_func_name
                    
        except Exception as e:
            print(f"Error reading {pt_file}: {e}")

    print("\n" + "="*60)
    print("DIRTY DATA ANALYSIS (Different Name, Same Content)")
    print("="*60)
    
    total = stats["total_scanned"]
    renamed = stats["renamed_dups_found"]
    exact = stats["exact_dups_ignored"]
    
    print(f"Total Functions Scanned      : {total:,}")
    print(f"Exact Duplicates (Ignored)   : {exact:,} (Same name, same content)")
    print(f"Renamed Duplicates (DIRTY)   : {renamed:,} (Diff name, same content)")
    
    if total > 0:
        dirty_rate = (renamed / total) * 100
        print(f"Dirty Data Rate              : {dirty_rate:.2f}%")
    
    print("-" * 60)
    
    if conflict_examples:
        print("Example Conflicts (First 10 detected):")
        for ex in conflict_examples:
            print(f" - {ex}")
    
    print("-" * 60)
    
    if dirty_proj_stats:
        print("Top 10 Projects with Renamed Duplicates:")
        sorted_projs = sorted(dirty_proj_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for proj, count in sorted_projs:
            print(f"{proj:<30} : {count} dirty functions")
    else:
        print("No renamed duplicates found. Data is clean regarding function renaming.")
        
    print("="*60)

if __name__ == "__main__":
    main()