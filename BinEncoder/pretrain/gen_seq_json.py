import json
import os




func_name_dir = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info\training_bert"
data_dir = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info\seq"


# 假设输入的 JSON 文件路径
input_file = r"F:\BinEncoder\dbs\Dataset-1\eval\info"  # 请替换成你的实际输入文件路径
# 假设输出的文件路径
output_file = r"F:\BinEncoder\dbs\Dataset-1\eval\seq"  # 请替换成你的实际输出文件路径
final = r"F:\BinEncoder\dbs\Dataset-1\eval\seq_final"
NOP_OPERAND = ' --- '
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
# # 转为seq
# # 读取 JSON 文件
# for dirpath, dirnames, filenames in os.walk(input_file):
#     for filename in filenames:
#         input_dir = os.path.join(input_file, filename)
#         print(f"正在处理 {input_dir}...")
#         with open(input_dir, 'r') as f:
#             data = json.load(f)
#
#             # 获取 "function Information" 部分
#             function_info = data.get("function Information", {})
#             output_dir = os.path.join(output_file, filename)
# # 打开输出文件以写入
#             with open(output_dir, 'w') as out_f:
#                 for address, details in function_info.items():
#                     # 将每个地址及其对应的数据写入输出文件
#                     out_f.write(json.dumps({address: details}, ensure_ascii=False) + '\n')
#
#
#
#
#
# for dirpath, dirnames, filenames in os.walk(output_file):
#     for filename in filenames:
#         func_dir = os.path.join(dirpath, filename)
#         print(f"正在处理 {func_dir}...")
#         transformed_data = {}
#         with open(func_dir, "r") as file:
#             # 逐行读取文件
#             for line in file:
#                 # 解析每行JSON数据
#                 original_data = json.loads(line.strip())
#                 for outer_key, outer_value in original_data.items():
#                     transformed_data[outer_key] = {
#                         "nverb": [],  # 初始化空列表
#                         "edges": outer_value["edges"]  # 直接复制 edges
#                     }
#                 # 遍历 nverb 中的键值对
#                     for inner_key, inner_value in outer_value["nverb"].items():
#                         inner_value = "".join(inner_value)
#                         inner = []
#                         if inner_value.startswith(" --- "):
#                             pcode_asm = inner_value[len(NOP_OPERAND):].strip()
#                             inner.append(NOP_OPERAND)
#                             a1 = pcode_asm.find(' ')
#                             if a1 != -1:
#                                 opc = pcode_asm[:a1]
#                                 inner.append(opc)
#                                 pcode_asm = pcode_asm[a1:].strip()
#                                 while len(pcode_asm) != 0:
#                                     pcode_asm = parse_nxopr(pcode_asm, inner)
#                             else:
#                                 opc = pcode_asm
#                                 inner.append(opc)
#                         else:
#                             pcode_asm = parse_nxopr(inner_value, inner)
#                             a = pcode_asm.find(' ')
#                             if a != -1:
#                                 opc = pcode_asm[:a].strip()
#                                 inner.append(opc)
#                                 pcode_asm = pcode_asm[a + 1:].strip()
#                                 while len(pcode_asm) != 0:
#                                     pcode_asm = parse_nxopr(pcode_asm, inner)
#                         inner = " ".join(inner)
#                         transformed_data[outer_key]["nverb"].append(inner)
#
#         out_func_dir = os.path.join(final, filename)
#         try:
#             with open(out_func_dir, "w") as file:
#                 for q, k in transformed_data.items():
#                     json.dump({q: k}, file, indent=None)
#                     file.write("\n")
#         except Exception as e:
#             print(f"写入文件 {out_func_dir} 时出错: {e}")
#             # 这里可以选择继续处理下一个文件，或者记录日志等操作



for dirpath, dirnames, filenames in os.walk(output_file):
    for filename in filenames:
        func_dir = os.path.join(dirpath, filename)
        print(f"正在处理 {func_dir}...")
        transformed_data = []
        with open(func_dir, "r") as file:
            # 逐行读取文件
            for line in file:
                # 解析每行JSON数据
                original_data = json.loads(line.strip())
                for outer_key, outer_value in original_data.items():
                # 遍历 nverb 中的键值对
                    line = []
                    for inner_key, inner_value in outer_value["nverb"].items():
                        inner_value = "".join(inner_value)
                        inner = []
                        if inner_value.startswith(" --- "):
                            pcode_asm = inner_value[len(NOP_OPERAND):].strip()
                            inner.append(NOP_OPERAND)
                            a1 = pcode_asm.find(' ')
                            if a1 != -1:
                                opc = pcode_asm[:a1]
                                inner.append(opc)
                                pcode_asm = pcode_asm[a1:].strip()
                                while len(pcode_asm) != 0:
                                    pcode_asm = parse_nxopr(pcode_asm, inner)
                            else:
                                opc = pcode_asm
                                inner.append(opc)
                        else:
                            pcode_asm = parse_nxopr(inner_value, inner)
                            a = pcode_asm.find(' ')
                            if a != -1:
                                opc = pcode_asm[:a].strip()
                                inner.append(opc)
                                pcode_asm = pcode_asm[a + 1:].strip()
                                while len(pcode_asm) != 0:
                                    pcode_asm = parse_nxopr(pcode_asm, inner)
                        inner = " ".join(inner)
                        line.append(inner)
                    str_line = ",".join(line)
                    transformed_data.append(str_line)
        out_func_dir = os.path.join(final, filename)
        with open(out_func_dir, "w") as file:
            for q in transformed_data:
                json.dump(q, file, indent=None)
                file.write("\n")




