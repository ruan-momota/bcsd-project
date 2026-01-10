import os
import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import glob
import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "asm_x64", "unrar")
OUTPUT_FILE = os.path.join(DATA_DIR, "outputs", "student_input.pt")
TEACHER_MODEL_ID = "hustcw/clap-asm"


def load_clap_tokenizer():
    print(f"Loading model: {config.TEACHER_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True)
    return tokenizer

def main():
    tokenizer = load_clap_tokenizer()
    tokenizer.model_max_length = 512

    # read asm .json files
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not json_files:
        print(f"Error: no asm .json files found in {INPUT_DIR}!")
        return

    json_files = json_files[:1] # for test!!!
    
    print(f"Start processing {len(json_files)} asm files...")

    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    metadata_list = []

    # traverse every asm .json file
    for json_path in tqdm(json_files, desc="Processing Files"):
        
        path_parts = json_path.split(os.sep)
        file_name = path_parts[-1]
        project_name = path_parts[-2]
        
        with open(json_path, 'r', encoding='utf-8') as f:
            functions_list = json.load(f)

        # batch data for current file
        batch_texts = []
        
        for func in functions_list:
            func_name = func.get('function_name')
            metadata_list.append({
                "project": project_name,
                "file_name": file_name,
                "func_name": func_name
            })

            func_input = func.copy()
            func_input.pop("function_name", None)
            batch_texts.append(func_input)

        for i in range(0, len(batch_texts), config.BATCH_SIZE):
            batch_data = batch_texts[i : i + config.BATCH_SIZE]
            
            inputs = tokenizer(
                batch_data, 
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )
            
            all_input_ids.append(inputs['input_ids'])
            all_attention_masks.append(inputs['attention_mask'])
            all_token_type_ids.append(inputs['token_type_ids'])

    print("Concatenating tensors...")
    if all_input_ids:
        final_input_ids = torch.cat(all_input_ids, dim=0)
        final_attention_masks = torch.cat(all_attention_masks, dim=0)
        final_token_type_ids = torch.cat(all_token_type_ids, dim=0)
        
        print(f"Total samples: {len(metadata_list)}")
        print(f"Input Ids Shape: {final_input_ids.shape}")
        
        save_data = {
            "student_input": {
                "input_ids": final_input_ids,
                "attention_mask": final_attention_masks,
                "token_type_ids": final_token_type_ids
            },
            "meta_data": metadata_list
        }

        print(f"Saving processed data to {OUTPUT_FILE} ...")
        torch.save(save_data, OUTPUT_FILE)
        print("Done！")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()