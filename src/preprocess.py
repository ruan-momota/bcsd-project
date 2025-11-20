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
        self.dataset = []  # 存储处理后的数据
    
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
                # 阈值设定：绝对值大于 65535 (0xFFFF) 视为大立即数
                if abs(val) > 0xFFFF: 
                    return "IMM"
            except ValueError:
                pass
        
        # 3. 检测十进制大数字
        if operand.lstrip('-').isdigit():
            try:
                if abs(int(operand)) > 65535:
                    return "IMM"
            except ValueError:
                pass
        
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
            cfg = proj.analyses.CFGFast(normalize=True)
            
            # 遍历所有函数
            for func in cfg.kb.functions.values():
                # 跳过非用户代码 (PLT表, 系统调用存根, 内存对齐填充等)
                if func.is_plt or func.is_simprocedure:
                    continue 
                
                func_instructions = [] # 存储该函数的完整指令列表
                
                # 遍历函数中的基本块
                for block in func.blocks:
                    # 遍历基本块中的指令 (使用 capstone)
                    for insn in block.capstone.insns:
                        # 1. 归一化得到 Token 列表 (例如: ['MOV', 'EAX', 'MEM'])
                        norm_tokens = self.normalize_instruction(insn)
                        
                        # 2. 更新词汇表 (Student 模型需要统计单个词)
                        self.vocab.update(norm_tokens)

                        # 3. 将 Token 重新拼接成指令字符串 (Teacher 模型需要指令结构)
                        # 例如: "MOV EAX MEM"
                        insn_str = " ".join(norm_tokens)
                        func_instructions.append(insn_str)
                
                # 过滤过短的函数 (少于 5 条指令)
                if len(func_instructions) < 5:
                    continue
                    
                # 添加到数据集
                # 注意：这里 Key 变成了 'instructions'
                self.dataset.append({
                    "project": project_name,
                    "binary": filename,
                    "func_name": func.name,
                    "instructions": func_instructions 
                })
                
        except Exception as e:
            # 捕获 angr 解析错误的个例，不中断整个流程
            print(f"\n[Warning] Error processing {filename}: {e}")

    def run(self):
        """
        主运行逻辑
        """
        print(f"开始遍历数据目录: {config.RAW_DATA_DIR}")

        # 调试用计数器 (如需测试少量数据，可取消下方注释)
        test_proj_count = 0
        
        # 遍历Dataset-1下的所有项目文件夹 (clamav, curl, etc.)
        projects = sorted(os.listdir(config.RAW_DATA_DIR))
        
        test1 = 0

        for project_name in projects:


            project_path = os.path.join(config.RAW_DATA_DIR, project_name)
            if not os.path.isdir(project_path):
                continue
            
            if test_proj_count >= 1: break # 调试用：只跑一个项目
            test_proj_count += 1

            print(f"正在处理项目: {project_name}")
            binary_files = [f for f in os.listdir(project_path) if not f.startswith('.')]
            
            # 使用 tqdm 显示进度条
            for bin_file in tqdm(binary_files, desc=f"Parsing {project_name}"):

                if test1 == 10:
                    break

                test1 += 1
                
                # 调试用：如果只想跑每个项目的前10个文件，可以取消下面两行注释
                # if binary_files.index(bin_file) >= 10: break 
                
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

        # 2. 保存处理后的序列数据
        print("保存训练数据...")
        with open(config.TRAIN_DATA_FILE, 'w') as f:
            # 使用jsonl格式 (每行一个json)
            for entry in self.dataset:
                f.write(json.dumps(entry) + '\n')
        print(f"处理完成，共提取 {len(self.dataset)} 个函数。")
        print(f"数据已保存至: {config.TRAIN_DATA_FILE}")

if __name__ == "__main__":
    processor = Preprocessor()
    processor.run()