import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FILE_PATH = os.path.join(DATA_DIR, "asm_x64", "unrar", "x64-clang-3.5-O2_unrar.json")


def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: cannot find {FILE_PATH}")
        return

    print(f"Reading: {FILE_PATH} ...")
    
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 检查数据结构
        if isinstance(data, list):
            count = len(data)
            print("-" * 30)
            print(f"✅ 文件解析成功")
            print(f"📊 该列表中共有 {count} 个元素 ({{...}})")
            print("-" * 30)
            
            # 可选：打印前几个看看
            if count > 0:
                print("第一个元素的内容预览:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False))
        else:
            print(f"⚠️ 文件格式正确，但不是列表，而是: {type(data)}")

    except json.JSONDecodeError:
        print("❌ 文件不是合法的 JSON 格式，无法解析。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    main()