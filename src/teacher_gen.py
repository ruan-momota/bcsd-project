import os
import json
import torch
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import config


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEACHER_INPUT_DIR = os.path.join(DATA_DIR, "asm_x64")
INPUT_DIR = os.path.join(TEACHER_INPUT_DIR, "unrar")
OUTPUT_FILE = os.path.join(DATA_DIR, "outputs", "teacher_embeddings.pt")


def load_clap_model(device):
    print(f"Loading model: {config.TEACHER_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True).to(device)
    model.eval()
    return tokenizer, model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")

    # load model
    tokenizer, model = load_clap_model(device)

    # read asm .json files
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not json_files:
        print(f"Error: no asm .json files found in {INPUT_DIR}!")
        return

    json_files = json_files[:1] # just for test!!!
    print(f"test: {os.path.basename(json_files[0])}")

    print(f"For test：only process {len(json_files)} files")

    all_file_embeddings = {}

    print(f"Start processing {len(json_files)} asm files...")

    # traverse every asm .json file
    for json_path in tqdm(json_files, desc="Processing Files"):

        filename = os.path.basename(json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            # a json file contains many functions
            functions_list = json.load(f)

            # original asm .json files should not have key function_name
            if isinstance(functions_list, list):
                for item in functions_list:
                    item.pop("function_name", None)

        file_embeddings = []

        for i in range(0, len(functions_list), config.BATCH_SIZE):
            # functions with batch_data numbers
            batch_data = functions_list[i : i + config.BATCH_SIZE]
            
            tokenizer.model_max_length = 1024

            inputs = tokenizer(
                batch_data, 
                padding='max_length',
                max_length=1024,
                return_tensors="pt"
            ).to(device)

            # check the shape of inputs
            if i == 0:
                print(f"check the shape of inputs...")
                print(f"type of inputs: {type(inputs)}")
                for key, value in inputs.items():
                    print(f"{key}: {value.shape}")

            with torch.no_grad():
                outputs = model(**inputs)
                file_embeddings.append(outputs.cpu())

        if file_embeddings:
            all_file_embeddings[filename] = torch.cat(file_embeddings, dim=0)
            print(f"embedding size of {filename}: {all_file_embeddings[filename].shape}")

    print(f"Saving embeddings to {OUTPUT_FILE} ...")
    torch.save(all_file_embeddings, OUTPUT_FILE)
    print("Done！")


if __name__ == "__main__":
    main()