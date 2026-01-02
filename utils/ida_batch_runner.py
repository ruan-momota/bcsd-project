import os
import subprocess
import glob


IDA_PATH = r"E:\ida\ida pro 7.6\ida pro 7.6\ida64.exe" 
SCRIPT_PATH = r"E:\ida\process_asm.py"
BINARY_DIR = r"E:\ida\Dataset-1\unrar"


def process_files():
    # 获取目录下所有文件 (这里假设没有后缀，或者你可以指定后缀如 *.exe)
    # 也可以使用 os.walk 递归遍历
    files = glob.glob(os.path.join(BINARY_DIR, "*"))
    
    for file_path in files:
        # 跳过生成的 json 文件或脚本本身
        if file_path.endswith(".json") or file_path.endswith(".py"):
            continue
            
        print(f"正在处理: {file_path} ...")
        
        # 构造命令
        # 格式: idat64 -A -c -S"脚本路径" "二进制文件路径"
        # 注意：-S 和脚本路径之间没有空格，且双引号是必须的，防止路径带空格
        cmd = [
            IDA_PATH,
            "-A",           # 自动模式
            "-c",           # 退出时删除 IDB 数据库（节省空间）
            f'-S{SCRIPT_PATH}', # 指定运行的脚本
            file_path       # 目标文件
        ]
        
        try:
            # 调用子进程执行
            subprocess.run(cmd, check=True)
            print(f"完成: {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"处理失败 {file_path}: {e}")
        except Exception as e:
            print(f"发生未知错误: {e}")

if __name__ == "__main__":
    process_files()