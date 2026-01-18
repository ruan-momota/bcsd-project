import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import config
import gc
from extract_asm import load_single_file_functions

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "asm_x64")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs", "teacher")

MAX_LENGTH = 128
BATCH_SIZE = 256

def load_clap_model(device):
    print(f"Loading model: {config.TEACHER_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    return tokenizer, model

def save_file_embeddings(data, proj_name, original_json_name):
    if not data:
        return

    save_dir = os.path.join(OUTPUT_DIR, str(BATCH_SIZE), proj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    pt_filename = original_json_name.replace(".json", ".pt")
    if not pt_filename.endswith(".pt"):
        pt_filename += ".pt"
        
    file_path = os.path.join(save_dir, pt_filename)
    torch.save(data, file_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")

    tokenizer, model = load_clap_model(device)
    tokenizer.model_max_length = MAX_LENGTH

    project_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    print(f"Found {len(project_dirs)} projects.")

    for project_name in tqdm(project_dirs, desc="Projects"):
        project_dir = os.path.join(INPUT_DIR, project_name)
        
        json_files = [f for f in os.listdir(project_dir) if f.endswith(".json")]

        for json_file in tqdm(json_files, desc=f"Inferencing {project_name}", leave=False):
            full_path = os.path.join(project_dir, json_file)
            file_funcs = load_single_file_functions(full_path)

            if not file_funcs:
                continue

            processed_samples = []
            asm_texts = [item['asm_text'] for item in file_funcs]
            
            for i in range(0, len(file_funcs), BATCH_SIZE):
                batch_asm = asm_texts[i : i + BATCH_SIZE]
                batch_metas = file_funcs[i : i + BATCH_SIZE]
                
                inputs = tokenizer(
                    batch_asm, 
                    padding='max_length',
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_embeddings = outputs.cpu()

                for j in range(len(batch_asm)):
                    meta = batch_metas[j]
                    sample = {
                        'proj_name': meta['proj_name'],
                        'file_name': meta['file_name'],
                        'func_name': meta['func_name'],
                        'teacher_embed': batch_embeddings[j]
                    }
                    processed_samples.append(sample)
            
            save_file_embeddings(processed_samples, project_name, json_file)
            
            del file_funcs
            del asm_texts
            del processed_samples
            
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone! All embeddings saved separately.")

if __name__ == "__main__":
    main()