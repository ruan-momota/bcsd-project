import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEACHER_OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "teacher_embeddings.pt")


# 预处理参数
MAX_SEQ_LEN = 512  # 函数指令序列的最大长度
MIN_COUNT = 5      # 最小词频，过滤极罕见token

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
MASK_TOKEN = "<MASK>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

TEACHER_MODEL_ID = "hustcw/clap-asm" 

BATCH_SIZE = 16
