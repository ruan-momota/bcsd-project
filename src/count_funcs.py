import os
import json
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "asm_x64")

def count_project_functions():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    project_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    project_dirs.sort()
    
    print(f"Found {len(project_dirs)} projects in {INPUT_DIR}\n")
    print(f"{'Project Name':<20} | {'Files':<8} | {'Raw Funcs':<12} | {'Valid Funcs':<12}")
    print("-" * 60)

    total_raw = 0
    total_valid = 0
    total_files = 0

    for project_name in project_dirs:
        project_dir = os.path.join(INPUT_DIR, project_name)
        json_files = [f for f in os.listdir(project_dir) if f.endswith(".json")]
        
        proj_raw_count = 0
        proj_valid_count = 0
        
        iterator = tqdm(json_files, desc=project_name, leave=False) if len(json_files) > 100 else json_files

        for json_file in iterator:
            file_path = os.path.join(project_dir, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    funcs = json.load(f)
                
                proj_raw_count += len(funcs)

                for func in funcs:
                    temp_func = func
                    temp_func.pop("function_name", None)

                    if len(temp_func) > 1:
                        proj_valid_count += 1

            except Exception as e:
                print(f"\nError reading {project_name}/{json_file}: {e}")

        print(f"{project_name:<20} | {len(json_files):<8} | {proj_raw_count:<12} | {proj_valid_count:<12}")
        
        total_raw += proj_raw_count
        total_valid += proj_valid_count
        total_files += len(json_files)

    print("-" * 60)
    print(f"{'TOTAL':<20} | {total_files:<8} | {total_raw:<12} | {total_valid:<12}")
    print(f"\nStats:")
    print(f"  - Total JSON Files: {total_files}")
    print(f"  - Total Raw Functions: {total_raw}")
    print(f"  - Total Valid Functions (Trainable): {total_valid}")

if __name__ == "__main__":
    count_project_functions()