import os
import subprocess
import glob


IDA_PATH = r"E:\ida\ida pro 7.6\ida pro 7.6\ida64.exe" 
SCRIPT_PATH = r"E:\ida\process_asm.py"
BINARY_DIR = r"E:\ida\Dataset-1\unrar"


def process_files():

    files = glob.glob(os.path.join(BINARY_DIR, "*"))
    
    for file_path in files:
        if file_path.endswith(".json") or file_path.endswith(".py"):
            continue
            
        print(f"processing: {file_path} ...")

        cmd = [
            IDA_PATH,
            "-A",        
            "-c",        
            f'-S{SCRIPT_PATH}', 
            file_path       
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"success: {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"failed {file_path}: {e}")
        except Exception as e:
            print(f"unknown error: {e}")

if __name__ == "__main__":
    process_files()