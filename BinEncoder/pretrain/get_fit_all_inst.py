import os
import json

# 输入和输出路径
input_json_path = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info"  # 替换为你的输入 JSON 文件路径
output_dir = r"E:\BinEncoder\dbs\Dataset-1\training\extracted"      # 替换为你的输出目录路径
#
# # 创建输出目录，如果不存在
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
#
# # 读取 JSON 文件
# for dir_path, dir_names, file_names in os.walk(input_json_path):
#     for file_name in file_names:
#         input_path = os.path.join(dir_path, file_name)  # 假设只有一个 JSON 文件
#         with open(input_path, 'r') as json_file:
#             data = json.load(json_file)
#
# # 处理每个文件的函数信息
# for file_name, functions in data["function Information"].items():
#     # 创建对应的输出文件路径
#     save_file = os.path.join(output_dir, f"{file_name}.txt")
#
#     # 打开保存文件以写入内容
#     with open(save_file, 'a') as fp_save:
#         for func_addr, details in functions.items():
#             # 提取 nverb 信息
#             if 'nverb' in details:
#                 for key, instructions in details['nverb'].items():
#                     sentence = ' '.join(instructions) + '\n'  # 将指令连接成字符串
#                     fp_save.write(f"nverb {key}: {sentence}")  # 记录 nverb 信息
#
#             # 提取 edges 信息
#             if 'edges' in details:
#                 for edge in details['edges']:
#                     edge_sentence = ' -> '.join(map(str, edge)) + '\n'  # 将边信息连接成字符串
#                     fp_save.write(f"edge: {edge_sentence}")  # 记录边的信息
#
# print(f"提取的信息已经保存到 {output_dir}")


# 创建输出目录，如果不存在
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 读取 JSON 文件
for dir_path, dir_names, file_names in os.walk(input_json_path):
    for file_name in file_names:
        if file_name.endswith('.json'):  # 确保只处理 JSON 文件
            input_path = os.path.join(dir_path, file_name)
            with open(input_path, 'r') as json_file:
                data = json.load(json_file)

            # 创建对应的输出文件路径，使用输入文件名称（去掉 .json 后缀）
            output_file_name = os.path.splitext(file_name)[0]  # 去掉 .json 后缀
            save_file = os.path.join(output_dir, f"{output_file_name}.txt")

            # 打开保存文件以写入内容
            with open(save_file, 'a') as fp_save:
                # 处理每个文件的函数信息
                for func_file_name, functions in data["function Information"].items():

                    # 提取 nverb 信息
                    if 'nverb' in functions:
                        for key, instructions in functions['nverb'].items():
                            sentence = ' '.join(instructions) + '\n'  # 将指令连接成字符串
                            fp_save.write(f"nverb {key}: {sentence}")  # 记录 nverb 信息

                    # 提取 edges 信息
                    if 'edges' in functions:
                        for edge in functions['edges']:
                            edge_sentence = ' -> '.join(map(str, edge)) + '\n'  # 将边信息连接成字符串
                            fp_save.write(f"edge: {edge_sentence}")  # 记录边的信息

        print(f"提取的信息已经保存到 {output_dir}")