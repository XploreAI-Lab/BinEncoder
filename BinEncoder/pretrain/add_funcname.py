import os
import json
import re


def parse_file_name(filename):
    """从文件名中提取信息，用于匹配对应的文件"""
    pattern = r"(.+?)_(.+?)_(.+?)_(.+?)_(.+?)_(.+?)\.elf"
    match = re.match(pattern, filename)
    if match:
        return match.groups()
    return None


def find_matching_files(path1, path2):
    """在两个路径下寻找对应的JSON文件"""
    matching_files = []

    # 获取第一个路径下所有JSON文件
    path1_files = [f for f in os.listdir(path1) if f.endswith('_cfg_summary.json')]

    # 获取第二个路径下所有JSON文件
    path2_files = [f for f in os.listdir(path2) if f.endswith('.json')]

    # 匹配文件
    for file1 in path1_files:
        file1_info = parse_file_name(file1)
        if file1_info:
            for file2 in path2_files:
                file2_info = parse_file_name(file2)
                if file2_info and file1_info == file2_info:
                    matching_files.append((os.path.join(path1, file1), os.path.join(path2, file2)))

    return matching_files


def enrich_json_files(file_pairs, out_dir):
    """根据第一个JSON文件的信息补充第二个JSON文件，并输出到指定目录"""
    # 确保输出目录存在
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file1_path, file2_path in file_pairs:
        # 读取第一个JSON文件
        with open(file1_path, 'r') as f1:
            data1 = json.load(f1)

        # 创建地址到函数名的映射
        addr_to_func_name = {}
        for key, funcs in data1.items():
            for func in funcs:
                # 将地址标准化为十六进制字符串格式
                start_ea = func["start_ea"].lower()
                addr_to_func_name[start_ea] = func["func_name"]

        # 读取第二个JSON文件（按行读取，每行是一个JSON对象）
        output_file_path = os.path.join(out_dir, os.path.basename(file2_path))
        enriched_lines = []

        with open(file2_path, 'r') as f2:
            for line_number, line in enumerate(f2, 1):
                try:
                    # 解析当前行的JSON对象
                    data2 = json.loads(line.strip())

                    # 补充函数名
                    if "func" in data2:
                        func_addr = data2["func"].lower()
                        # 直接匹配
                        if func_addr in addr_to_func_name:
                            data2["func_name"] = addr_to_func_name[func_addr]
                        else:
                            # 尝试不同的格式匹配
                            for addr in addr_to_func_name:
                                # 移除"0x"前缀再比较
                                if addr.replace("0x", "") == func_addr.replace("0x", ""):
                                    data2["func_name"] = addr_to_func_name[addr]
                                    break

                    # 将补充后的JSON对象添加到结果列表
                    enriched_lines.append(json.dumps(data2))
                except json.JSONDecodeError:
                    print(f"警告: 文件 {file2_path} 的第 {line_number} 行不是有效的JSON，已跳过")
                    # 保留原始行
                    enriched_lines.append(line.strip())

        # 将所有处理后的行写入输出文件
        with open(output_file_path, 'w') as f_out:
            for line in enriched_lines:
                f_out.write(line + '\n')

        print(f"已处理文件: {os.path.basename(file2_path)} → {output_file_path} (共 {len(enriched_lines)} 行)")


def main():
    # 用户输入路径
    path1 = r"C:\Users\tianh\Desktop\eval\disasm"
    path2 = r"F:\BinEncoder\dbs\Dataset-1\eval\embeddings"
    out_dir = r"F:\BinEncoder\dbs\Dataset-1\eval\name_emb"

    # 找到匹配的文件对
    matching_files = find_matching_files(path1, path2)

    if not matching_files:
        print("未找到匹配的文件对!")
        return

    print(f"找到 {len(matching_files)} 对匹配的文件")

    # 补充JSON文件并输出到指定目录
    enrich_json_files(matching_files, out_dir)

    print(f"处理完成! 所有文件已输出到: {out_dir}")


if __name__ == "__main__":
    main()