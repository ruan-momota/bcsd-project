import ijson
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
FILE_PATH = os.path.join(DATA_ROOT, "asm_x64", "unrar", "x64-clang-3.5-O0_unrar.json")


def count_elements_large_file(file_path):

    count = 0
    start_time = time.time()
    
    print(f"Opening : {file_path}")
    print("Started, it takes time...")

    try:
        # 注意：ijson 建议以二进制模式 'rb' 打开文件
        with open(file_path, 'rb') as f:
            # 'item' 表示我们要遍历根数组中的每一个项目
            # ijson.items 返回的是一个生成器，每次只产生一个对象，不占用多余内存
            elements = ijson.items(f, 'item')
            
            for _ in elements:
                count += 1
                
                # 可选：每处理 10,000 个元素打印一次进度，避免觉得程序卡死
                if count % 10000 == 0:
                    print(f"\rScanned {count} elements...", end="", flush=True)

        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nCompleted！")
        print(f"Total elements: {count}")
        print(f"Took time: {duration:.2f} seconds")
        return count

    except FileNotFoundError:
        print(f"\nERROR, cannot found file: '{file_path}'")
    except Exception as e:
        print(f"\nERROR, unknown problem: {e}")


if __name__ == "__main__":

    target_file = FILE_PATH
    count_elements_large_file(target_file)