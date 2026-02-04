import json
import os

def load_project_functions(project_path):
    results = []
    if not os.path.exists(project_path):
        return results

    proj_name = os.path.basename(project_path)
    json_files = [f for f in os.listdir(project_path) if f.endswith(".json")]

    for file_name in json_files:
        file_path = os.path.join(project_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                funcs = json.load(f)
            
            for func in funcs:
                func_name = func.pop("function_name", None)
                if len(func) <= 5: 
                    continue

                results.append({
                    "proj_name": proj_name,
                    "file_name": file_name,
                    "func_name": func_name,
                    "asm_text": func
                })

        except Exception as e:
            print(f"Error reading {proj_name}: {file_name}: {e}")
        
    return results

def load_single_file_functions(file_path):
    results = []
    if not os.path.exists(file_path):
        return results

    file_name = os.path.basename(file_path)
    proj_name = os.path.basename(os.path.dirname(file_path))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            funcs = json.load(f)
        
        for func in funcs:
            func_name = func.pop("function_name", None)
            if len(func) <= 5: 
                continue

            results.append({
                "proj_name": proj_name,
                "file_name": file_name,
                "func_name": func_name,
                "asm_text": func
            })

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return results