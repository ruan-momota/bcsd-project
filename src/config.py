import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEACHER_OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "teacher_embeddings.pt")

TEACHER_MODEL_ID = "hustcw/clap-asm" 

BATCH_SIZE = 16
