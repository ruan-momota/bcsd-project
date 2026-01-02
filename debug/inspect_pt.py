import torch
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

file_path = os.path.join(DATA_DIR, "processed", "teacher_embeddings.pt")

print(f"正在加载文件: {file_path}")
data = torch.load(file_path, map_location='cpu')

print("-" * 30)
print(f"顶层数据类型: {type(data)}")

if isinstance(data, dict):
    print(f"这是一个字典，包含 {len(data)} 个文件的 Embedding。")
    print(f"文件列表 (Keys): {list(data.keys())}")
    
    # === 获取第一个文件的 Embedding 来查看详情 ===
    first_filename = list(data.keys())[0]
    embedding_tensor = data[first_filename]
    
    print("-" * 30)
    print(f"正在检查第一个文件: {first_filename}")
    
    if isinstance(embedding_tensor, torch.Tensor):
        print(f"张量形状 (Shape): {embedding_tensor.shape}")
        print(f"   -> {embedding_tensor.shape[0]} 个函数")
        print(f"   -> {embedding_tensor.shape[1]} 维向量 (Teacher Embedding 维度)")
        print(f"数据类型 (Dtype): {embedding_tensor.dtype}")
        print("\n前 5 个函数的向量数据预览:")
        print(embedding_tensor[:5])
    else:
        print(f"异常: 字典的值不是 Tensor，而是 {type(embedding_tensor)}")

elif isinstance(data, torch.Tensor):
    print(f"文件是纯张量。形状: {data.shape}")
    print(data[:5])