import json
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DIR = os.path.join(DATA_DIR, "bcsd_benchmark_5", "val", 'val_ground_truth.json')

def analyze_answer_counts(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    counts = [len(answers) for query_id, answers in data.items()]
    
    if not counts:
        return 0, 0, 0
        
    min_count = np.min(counts)
    max_count = np.max(counts)
    avg_count = np.mean(counts)
    
    return min_count, max_count, avg_count

file_path = DIR
min_val, max_val, avg_val = analyze_answer_counts(file_path)

print(f"min: {min_val}")
print(f"max: {max_val}")
print(f"mean: {avg_val}")