import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import gc
import config
from extract_asm import load_single_file_functions

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "asm_x64")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs", "student")

BATCH_SIZE = 32
TOKENIZER_LEN = 128

def load_clap_tokenizer():
    print(f"Loading model: {config.TEACHER_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True)
    return tokenizer

def save_file_data(data, proj_name, original_json_name):
    if not data:
        return

    save_dir = os.path.join(OUTPUT_DIR, str(TOKENIZER_LEN), proj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    pt_filename = original_json_name.replace(".json", ".pt")
    if not pt_filename.endswith(".pt"):
        pt_filename += ".pt"
        
    file_path = os.path.join(save_dir, pt_filename)
    torch.save(data, file_path)

def main():
    tokenizer = load_clap_tokenizer()
    tokenizer.model_max_length = TOKENIZER_LEN

    project_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    print(f"Found {len(project_dirs)} projects.")

    for project_name in tqdm(project_dirs, desc="Projects"):
        project_dir = os.path.join(INPUT_DIR, project_name)
        json_files = [f for f in os.listdir(project_dir) if f.endswith(".json")]
        
        for json_file in tqdm(json_files, desc=f"Processing {project_name}", leave=False):
            full_path = os.path.join(project_dir, json_file)
            file_funcs = load_single_file_functions(full_path)
            
            if not file_funcs:
                continue

            file_processed_buffer = []
            current_texts = [item['asm_text'] for item in file_funcs]
            
            for i in range(0, len(file_funcs), BATCH_SIZE):
                batch_texts = current_texts[i : i + BATCH_SIZE]
                batch_metas = file_funcs[i : i + BATCH_SIZE]
                
                inputs = tokenizer(
                    batch_texts, 
                    padding='max_length',
                    max_length=TOKENIZER_LEN,
                    return_tensors="pt"
                )
                
                for j in range(len(batch_texts)):
                    meta = batch_metas[j]
                    sample = {
                        'proj_name': meta['proj_name'],
                        'file_name': meta['file_name'],
                        'func_name': meta['func_name'],
                        'student_input': {
                            "input_ids": inputs['input_ids'][j],
                            "attention_mask": inputs['attention_mask'][j],
                            "token_type_ids": inputs['token_type_ids'][j]
                        }
                    }
                    file_processed_buffer.append(sample)
            
            save_file_data(file_processed_buffer, project_name, json_file)
            
            del file_funcs
            del current_texts
            del file_processed_buffer

        gc.collect()

    print("\nProcessing complete. All files saved.")

if __name__ == "__main__":
    main()