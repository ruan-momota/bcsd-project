import os
import torch
import json
import hashlib
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "outputs", "teacher", "128")
BLOCKLIST_FILE = os.path.join(DATA_DIR, "outputs", "blocklist128.json")
EXCLUDE_PROJECTS = ["z3"]

def get_embedding_fingerprint(tensor):
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Input dir not found: {INPUT_DIR}")
        return

    fingerprint_map = {}
    blocklist = []
    
    stats = {
        "total": 0, 
        "kept": 0, 
        "dirty_dup": 0,   
        "exact_dup": 0     
    }
    project_stats = {}
    pt_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        # if "z3" in root.split(os.sep): continue
        for f in files:
            if f.endswith(".pt"):
                pt_files.append(os.path.join(root, f))

    print(f"Scanning {len(pt_files)} files...")

    for pt_file in tqdm(pt_files, desc="Building Blocklist"):
        try:
            data = torch.load(pt_file, map_location='cpu')
            if not data: continue

            for item in data:
                stats["total"] += 1
                
                proj_name = item['proj_name']
                unique_key = f"{proj_name}|{item['file_name']}|{item['func_name']}"
                fp = get_embedding_fingerprint(item['teacher_embed'])

                if fp not in fingerprint_map:
                    # first unique func occur
                    fingerprint_map[fp] = item['func_name']
                    stats["kept"] += 1
                    project_stats[proj_name] = project_stats.get(proj_name, 0) + 1
                else:
                    blocklist.append(unique_key)
                    original_name = fingerprint_map[fp]
                    if item['func_name'] != original_name:
                        stats["dirty_dup"] += 1
                    else:
                        stats["exact_dup"] += 1
                        
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")

    print(f"\nWriting {len(blocklist)} entries to {BLOCKLIST_FILE}...")
    with open(BLOCKLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(blocklist, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 40)
    print(f"Global Statistics:")
    print(f"  Total Processed:    {stats['total']}")
    print(f"  Kept (Unique):      {stats['kept']}")
    print(f"  Blocked (Dirty):    {stats['dirty_dup']} (Same Body, Diff Name)")
    print(f"  Blocked (Exact):    {stats['exact_dup']} (Same Body, Same Name)")
    
    dup_rate = (stats['dirty_dup'] + stats['exact_dup']) / stats['total'] if stats['total'] > 0 else 0
    print(f"  Total Filtered Rate: {dup_rate:.2%}")
    
    print("-" * 40)
    print(f"Kept Functions by Project (Descending):")
    sorted_projects = sorted(project_stats.items(), key=lambda x: x[1], reverse=True)
    for proj, count in sorted_projects:
        print(f"  {proj:<20}: {count}")
    print("=" * 40)

if __name__ == "__main__":
    main()