import torch
import config  # 导入你的配置，为了获取路径

# 1. 设定文件路径 (或者直接写绝对路径)
file_path = config.TEACHER_EMBEDDINGS_FILE
print(f"正在加载文件: {file_path}")

# 2. 加载 .pt 文件
# map_location='cpu' 保证即使你没有显卡也能查看数据
data = torch.load(file_path, map_location='cpu')

# 3. 查看数据类型和形状
print("-" * 30)
print(f"数据类型: {type(data)}")

if isinstance(data, torch.Tensor):
    print(f"张量形状 (Shape): {data.shape}")
    print(f"数据类型 (Dtype): {data.dtype}")
    print("-" * 30)
    print("前 5 行向量数据预览:")
    print(data[:5]) # 打印前5个函数的向量
else:
    print("文件内容不是纯张量，可能是模型权重字典 (state_dict)。")
    print(data.keys())