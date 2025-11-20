import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------
# 1. 加载 CLAP-ASM
# -----------------------
MODEL_NAME = "whfzy/clap-asm"   # 示例名称，换成你使用的CLAP-ASM模型名

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -----------------------
# 2. 读取 Dataset-1 所有 ASM
# -----------------------
ASM_DIR = "data/functions/"
asm_files = sorted(os.listdir(ASM_DIR))

functions = {}
for fname in asm_files:
    fid = fname.split(".")[0]
    with open(os.path.join(ASM_DIR, fname), "r") as f:
        text = f.read()
    functions[fid] = text

print(f"Loaded {len(functions)} functions.")

# -----------------------
# 3. 生成 teacher embeddings
# -----------------------
teacher_vectors = []
function_ids = list(functions.keys())

with torch.no_grad():
    for fid in function_ids:
        asm_text = functions[fid]

        # 输入 CLAP-ASM
        inputs = tokenizer(
            asm_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)

        # 取 CLS embedding
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        teacher_vectors.append(emb)

teacher_vectors = np.array(teacher_vectors)

# -----------------------
# 4. 保存 teacher_target
# -----------------------
np.save("teacher_target.npy", teacher_vectors)
json.dump(function_ids, open("teacher_ids.json", "w"))

print("Saved teacher_target.npy")
print("Shape:", teacher_vectors.shape)
