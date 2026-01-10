import json
import os

def filter_single_element_asm(results):
    filtered_results = []
    for item in results:
        if len(item['asm_text']) != 1:
            filtered_results.append(item)
    return filtered_results

def load_project_functions(project_path):
    """
    load all json files under the path
    return: [{'proj_name': str, 
              'file_name': str, 
              'func_name': str, 
              'asm_text': str}]
    """
    results = []

    if not os.path.exists(project_path):
        return results

    proj_name = os.path.basename(project_path)

    for file_name in os.listdir(project_path):
        if not file_name.endswith(".json"):
            continue
            
        file_path = os.path.join(project_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                funcs = json.load(f)
            
            for func in funcs:
                func_name = func.get("function_name")
                func.pop("function_name", None)
                results.append({
                    "proj_name": proj_name,
                    "file_name": file_name,
                    "func_name": func_name,
                    "asm_text": func
                })

        except Exception as e:
            print(f"Error reading {file_name}: {e}")
        
        results = filter_single_element_asm(results)
        
    return results