import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "Dataset-1")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 确保输出目录存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 预处理参数
VOCAB_SIZE = 5000  # 词汇表大小预估
MAX_SEQ_LEN = 512  # 函数指令序列的最大长度
MIN_COUNT = 5      # 最小词频，过滤极罕见token

# 特殊Token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
MASK_TOKEN = "<MASK>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

# 文件保存路径
VOCAB_FILE = os.path.join(PROCESSED_DATA_DIR, "vocab.json")
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "train_data.json")

# 这是我们将要生成的纯ID文件
TRAIN_ID_FILE = os.path.join(PROCESSED_DATA_DIR, "train_ids.json")

# Teacher模型配置
# 注意：如果 CLAP-ASM 是你本地微调过的模型，请将此处改为本地文件夹路径
# 这里我暂且用 'microsoft/codebert-base' 作为占位符，请替换为你实际使用的 CLAP-ASM 模型 ID 或路径
TEACHER_MODEL_ID = "microsoft/codebert-base" 

# 知识向量保存路径
# 我们将生成的向量矩阵保存为 PyTorch 的 .pt 文件，加载速度最快
TEACHER_EMBEDDINGS_FILE = os.path.join(PROCESSED_DATA_DIR, "teacher_embeddings.pt")

# 推理批次大小 (根据你的显存调整，显存大可以设为 32 或 64)
BATCH_SIZE = 16