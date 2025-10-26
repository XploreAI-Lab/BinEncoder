import os
import json
import glob
from tqdm import tqdm
import random
from collections import defaultdict


def process_directory(input_dir, output_file):
    # 第一阶段：收集所有函数数据（带进度条）
    print("正在扫描目录并收集函数数据...")
    func_dict = defaultdict(list)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    # 使用tqdm显示文件读取进度
    for file_path in tqdm(json_files, desc="读取JSON文件"):

        for line in open(file_path, 'r'):
            data = json.loads(line)
            for func_hash, func_data in data.items():
                # 提取函数特征（nverb内容）
                features = [line.strip() for line in func_data.get("nverb", [])]
                if features:
                    func_dict[func_hash].append({
                        "path": file_path,
                        "features": features
                    })


    # 第二阶段：生成三元组（带进度条）
    print("\n生成训练三元组...")
    triplets = []

    # 准备函数哈希列表用于负采样
    func_hashes = list(func_dict.keys())

    # 使用tqdm显示处理进度
    for func_hash, samples in tqdm(func_dict.items(), desc="处理函数"):
        if len(samples) < 2:
            continue  # 需要至少2个样本来创建正样本对

        # 生成正样本对
        for i in range(len(samples)):
            anchor = samples[i]

            # 寻找正样本（相同函数的不同实例）
            positive = samples[(i + 1) % len(samples)]

            # 生成负样本（不同函数）
            neg_hash = func_hash
            attempts = 0
            while neg_hash == func_hash and attempts < 10:
                neg_hash = random.choice(func_hashes)
                attempts += 1

            if neg_hash == func_hash:
                continue  # 找不到合适的负样本

            negative_samples = func_dict[neg_hash]
            if not negative_samples:
                continue

            negative = random.choice(negative_samples)

            # 构建三元组
            triplets.append({
                "anchor": {
                    "path": anchor["path"],
                    "features": anchor["features"]
                },
                "positive": {
                    "path": positive["path"],
                    "features": positive["features"]
                },
                "negative": {
                    "path": negative["path"],
                    "features": negative["features"]
                }
            })

    # 第三阶段：保存结果（带进度条）
    print("\n保存训练数据...")
    with open(output_file, 'w') as f:
        for triplet in tqdm(triplets, desc="写入文件"):
            json_line = json.dumps(triplet)
            f.write(json_line + "\n")

    print(f"\n完成！已生成 {len(triplets)} 个训练三元组，保存至 {output_file}")


if __name__ == "__main__":
    input_directory = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info\final"  # 修改为你的目录
    output_filename = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info\triplet\triplets.json"
    process_directory(input_directory, output_filename)

