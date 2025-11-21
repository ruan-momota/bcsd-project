# debug_clap.py
import inspect
from transformers import AutoTokenizer

# 1. 加载 Tokenizer (允许远程代码)
print("正在加载 Tokenizer 代码...")
try:
    tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
except Exception as e:
    print(f"加载失败: {e}")
    exit()

# 2. 获取 __call__ 方法的源代码
print("\n" + "="*40)
print("CLAP-ASM Tokenizer.__call__ 源码分析")
print("="*40)

try:
    # 获取 __call__ 函数的源码
    source_code = inspect.getsource(tokenizer.__call__)
    
    # 打印前 50 行，通常输入格式检查就在开头
    lines = source_code.split('\n')
    for i, line in enumerate(lines[:50]):
        print(f"{i+1:03d}: {line}")
        
except Exception as e:
    print(f"无法获取源码: {e}")

print("\n" + "="*40)
print("请查看上方代码，寻找类似 'functions[0]['key_name']' 的字样")
print("="*40)