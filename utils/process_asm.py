# https://github.com/Hustcw/CLAP/tree/main/scripts

import idautils # type: ignore
import idaapi # type: ignore
import idc # type: ignore
import re
import json
import os

loc_pattern = re.compile(r' (loc|locret)_(\w+)')   
self_pattern = re.compile(r'\$\+(\w+)')     

def rebase(asm_dict):
    index = 1
    rebase_assembly = {}

    addrs = list(sorted(list(asm_dict.keys())))

    for addr in addrs:
        inst = asm_dict[addr] # inst是汇编指令字符串，如call sub_401020, mov eax, 1
        if inst.startswith('j'): # 筛选跳转指令，jmp,jnz,je,jg
            loc = loc_pattern.findall(inst) # jnz loc_401020会匹配出('loc','401020')
            for prefix, target_addr in loc:
                try:
                    target_instr_idx = addrs.index(int(target_addr, 16)) + 1
                except Exception:
                    continue
                asm_dict[addr] = asm_dict[addr].replace(
                    f' {prefix}_{target_addr}', f' INSTR{target_instr_idx}')
            self_m = self_pattern.findall(inst)
            for offset in self_m:
                target_instr_addr = addr + int(offset, 16)
                try:
                    target_instr_idx = addrs.index(target_instr_addr)
                    asm_dict[addr] = asm_dict[addr].replace(
                        f'$+{offset}', f'INSTR{target_instr_idx}')
                except:
                    continue
        rebase_assembly[str(index)] = asm_dict[addr]
        index += 1

    # 返回的字典：{'1': 'mov eax, 1', '2': '...'}
    # id对应一个instruction
    return rebase_assembly

# 获取指定函数地址ea下的所有指令，并调用rebase进行清洗
def get_assembly_by_ea(ea):
    instGenerator = idautils.FuncItems(ea)
    raw_assembly = {}
    for inst in instGenerator:  # inst应该是每个指令的地址？
        raw_assembly[inst] = idc.GetDisasm(inst)
        
    rebased_assembly = rebase(raw_assembly) 
    return rebased_assembly

if __name__ == '__main__':
    idc.auto_wait()
    binary_abs_path = idc.get_input_file_path()

    # ================= 修改开始 =================
    
    # 1. 定义你想要存放结果的绝对路径目录 (请修改为你电脑上的实际路径)
    # 建议使用 r"" 原始字符串格式，防止 Windows 路径中的反斜杠被转义
    target_output_dir = r"E:\ida\outputs\unrar" 
    
    # 2. (可选) 检查目录是否存在，不存在则自动创建，防止报错
    if not os.path.exists(target_output_dir):
        try:
            os.makedirs(target_output_dir)
        except OSError:
            pass # 如果并发创建可能报错，这里简单忽略
            
    # 3. 从完整路径中提取文件名 (例如从 "C:\Malware\virus.exe" 提取出 "virus.exe")
    file_name = os.path.basename(binary_abs_path)
    
    # 4. 拼接绝对路径和文件名
    # os.path.join 会自动处理 Windows(\) 和 Linux(/) 的路径分隔符差异
    output_path = os.path.join(target_output_dir, file_name + '.json')
    
    # ================= 修改结束 =================

    function_list = idautils.Functions()
    result = []
    for func_ea in function_list:
        # 得到转换为INSTR的指令，一个function_data应该对应一个函数
        function_data = get_assembly_by_ea(func_ea)
        result.append(function_data)
    json.dump(result, open(output_path, 'w'))
    idc.qexit(0)
