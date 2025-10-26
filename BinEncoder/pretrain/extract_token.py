import os
import re

NOP_OPERAND = '  ---'
add_vocal_list = list()
def parse_nxopr(pcode_asm, add_vocal_list):
    s = pcode_asm.find('(')
    e = pcode_asm.find(')')
    varnode = pcode_asm[s:e+1]
    varnode_parts = varnode.split(',')
    varnode_parts[1] = " addr"
    varnode = ','.join(varnode_parts)
    add_vocal_list.append(varnode)
    pcode_asm = pcode_asm[e+1:].strip()
    return pcode_asm


def parse_pcode(block_insts):
    if block_insts.startswith('nverb'):
        block_insts = block_insts.split(":")[1]
        if block_insts.startswith('  ---  '):
            pcode_asm = block_insts[len(NOP_OPERAND) + 1:].strip()
            a1 = pcode_asm.find(' ')
            if a1 != -1:
                opc = pcode_asm[:a1]
                add_vocal_list.append(opc)
                pcode_asm = pcode_asm[a1:].strip()
                while len(pcode_asm) != 0:
                    pcode_asm = parse_nxopr(pcode_asm, add_vocal_list)
            else:
                opc = pcode_asm
                add_vocal_list.append(opc)
        else:
            pcode_asm = parse_nxopr(block_insts, add_vocal_list)
            a = pcode_asm.find(' ')
            if a != -1:
                opc = pcode_asm[:a].strip()
                add_vocal_list.append(opc)
                pcode_asm = pcode_asm[a+1:].strip()
                while len(pcode_asm) != 0:
                    pcode_asm = parse_nxopr(pcode_asm, add_vocal_list)


def remove_duplicates_from_list(lines):
    unique_lines = []
    seen = set()

    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    return unique_lines

if __name__ == '__main__':
    st_instc_dir = r"E:\BinEncoder\dbs\Dataset-1\training\extracted"  # 输入目录
    add_vocal = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_token\add_vocal.txt"  # 输出文件路径


    for instc in os.listdir(st_instc_dir):
        st_instc_path = os.path.join(st_instc_dir, instc)  # 使用 os.path.join 生成路径

        with open(st_instc_path, 'r') as fp:
            read_tokens = fp.readlines()

            for block_insts in read_tokens:
                parse_pcode(block_insts)

    add_vocal_list = remove_duplicates_from_list(add_vocal_list)
    add_vocal_list.append(" --- ")

    # 输出每个拆分内容为单独一行
    with open(add_vocal, "w") as ap:
        for add_vocal_l in add_vocal_list:
            # 如果指令是特定格式，比如以括号开头或包含函数名，则单独处理
            ap.write(f"{add_vocal_l}\n")  # 直接写入，一行一个元素