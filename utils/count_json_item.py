import ijson
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
FILE_PATH = os.path.join(DATA_ROOT, "bcsd_benchmark", "bcsd_pool.json")


def count_elements_large_file(file_path):

    count = 0
    start_time = time.time()
    
    print(f"Opening : {file_path}")
    print("Started, it takes time...")

    try:
        with open(file_path, 'rb') as f:
            elements = ijson.items(f, 'item')
            
            for _ in elements:
                count += 1
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