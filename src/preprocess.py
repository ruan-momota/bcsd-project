# src/preprocess.py
import os
import json
import angr
import re
from tqdm import tqdm
from collections import Counter
import config

class Preprocessor:
    def __init__(self):
        self.vocab = Counter()
        self.dataset = []  # 存储处理后的数据: [{"func_name": str, "tokens": [str], "project": str}]
    
    def normalize_operand(self, operand):
        """
        归一化操作数的逻辑：
        - 内存引用 -> MEM
        - 大的立即数 -> IMM
        - 小常数/寄存器 -> 保留
        """
        # 1. 检测内存引用 (通常包含括号 [] 或 sp/bp 偏移)
        if '[' in operand or ']' in operand:
            return "MEM"
        
        # 2. 检测十六进制大立即数 (如 0x104f)
        if operand.startswith("0x") or operand.startswith("-0x"):
            try:
                val = int(operand, 16)
                # 阈值设定：绝对值大于 65535 (0xFFFF) 视为大立即数，或者是地址
                if abs(val) > 0xFFFF: 
                    return "IMM"
            except ValueError:
                pass
        
        # 3. 检测十进制大数字
        if operand.lstrip('-').isdigit():
            if abs(int(operand)) > 65535:
                return "IMM"
        
        # 4. 其他保留 (寄存器、小常数)
        return operand

    def normalize_instruction(self, insn):
        """
        将angr/capstone的指令对象转换为归一化的token列表
        """
        # 获取助记符 (Mnemonic)
        mnemonic = insn.mnemonic
        op_str = insn.op_str
        
        tokens = [mnemonic]
        
        if op_str:
            # 分割操作数，通常用逗号分隔
            operands = [op.strip() for op in op_str.split(',')]
            for op in operands:
                norm_op = self.normalize_operand(op)
                tokens.append(norm_op)
                
        return tokens

    def process_binary(self, file_path, project_name):
        """
        使用angr处理单个二进制文件
        """
        filename = os.path.basename(file_path)
        try:
            # 加载二进制，禁用自动加载库以加快速度
            proj = angr.Project(file_path, auto_load_libs=False, load_debug_info=False)
            
            # 生成控制流图 (CFG) 来识别函数
            # CFGFast 比较快，但可能不如 Emulated 准确，对于大规模数据通常够用
            cfg = proj.analyses.CFGFast(normalize=True)
            
            functions = cfg.kb.functions
            
            for func_addr, func in functions.items():
                if func.is_plt or func.is_simprocedure:
                    continue # 跳过非用户代码
                
                func_tokens = []
                
                # 遍历函数中的基本块
                for block in func.blocks:
                    # 遍历基本块中的指令 (使用 capstone)
                    for insn in block.capstone.insns:
                        # 归一化并添加到列表
                        norm_tokens = self.normalize_instruction(insn)
                        func_tokens.extend(norm_tokens)
                
                # 过滤过短的函数
                if len(func_tokens) < 5:
                    continue
                    
                # 添加到数据集
                self.dataset.append({
                    "project": project_name,
                    "binary": filename,
                    "func_name": func.name,
                    "tokens": func_tokens
                })
                
                # 更新词汇统计
                self.vocab.update(func_tokens)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def run(self):
        """
        主运行逻辑
        """
        print(f"开始遍历数据目录: {config.RAW_DATA_DIR}")

        # for test use
        test = 0
        test1 = 0
        
        # 遍历Dataset-1下的所有项目文件夹 (clamav, curl, etc.)
        for project_name in os.listdir(config.RAW_DATA_DIR):

            if test == 1: break
            test+=1

            project_path = os.path.join(config.RAW_DATA_DIR, project_name)
            if not os.path.isdir(project_path):
                continue
            
            print(f"正在处理项目: {project_name}")
            binary_files = [f for f in os.listdir(project_path) if not f.startswith('.')]
            
            for bin_file in tqdm(binary_files):

                # for test use
                if test1 == 10: break
                test1+=1

                bin_path = os.path.join(project_path, bin_file)
                self.process_binary(bin_path, project_name)
        
        self.save_data()

    def save_data(self):
        # 1. 构建并保存词汇表
        print("构建词汇表...")
        # 过滤低频词
        final_vocab = {k: v for k, v in self.vocab.items() if v >= config.MIN_COUNT}
        # 添加特殊Token
        word2id = {token: i for i, token in enumerate(config.SPECIAL_TOKENS)}
        current_id = len(config.SPECIAL_TOKENS)
        
        # 按频率排序添加
        for word, _ in self.vocab.most_common(config.VOCAB_SIZE):
            if word not in word2id and word in final_vocab:
                word2id[word] = current_id
                current_id += 1
        
        # 保存 vocab
        with open(config.VOCAB_FILE, 'w') as f:
            json.dump(word2id, f, indent=2)
        print(f"词汇表已保存，大小: {len(word2id)}")

        # 2. 保存处理后的序列数据 (此时还是文本token，后续Dataset类会转为ID)
        print("保存训练数据...")
        with open(config.TRAIN_DATA_FILE, 'w') as f:
            # 使用jsonl格式 (每行一个json) 更节省内存且方便读取
            for entry in self.dataset:
                f.write(json.dumps(entry) + '\n')
        print(f"处理完成，共提取 {len(self.dataset)} 个函数。")

if __name__ == "__main__":
    processor = Preprocessor()
    processor.run()