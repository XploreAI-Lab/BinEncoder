import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_embeddings(input_dir):
    """从目录中的JSON文件加载嵌入"""
    ebds = []
    # 用于跟踪每个函数的不同编译配置的嵌入
    func_embeddings = {}

    for fname in tqdm(os.listdir(input_dir), desc='加载json'):
        if not fname.endswith('_embeddings.json'):
            continue

        # 解析文件名以获取编译器/架构信息
        parts = fname.split('_')
        if len(parts) >= 5:
            compiler_ver = parts[1]
            arch = parts[2]
            bits = parts[3]
            opt = parts[4]
            type_key = '_'.join([compiler_ver, arch, bits, opt])

            load_path = os.path.join(input_dir, fname)
            try:
                with open(load_path, 'r') as ff:
                    for line in ff:
                        data = json.loads(line.strip())
                        if "func_name" in data and "embedding" in data:
                            # 提取实际嵌入向量（移除嵌套列表）
                            if isinstance(data["embedding"], list) and len(data["embedding"]) > 0:
                                if isinstance(data["embedding"][0], list):
                                    embedding = data["embedding"][0]  # 获取内部列表
                                else:
                                    embedding = data["embedding"]  # 已经是平面列表

                                # 使用函数名作为主键
                                func_name = data["func_name"]

                                # 如果函数名不存在，创建新条目
                                if func_name not in func_embeddings:
                                    func_embeddings[func_name] = {}

                                # 存储特定编译配置的嵌入
                                func_embeddings[func_name][type_key] = embedding

            except Exception as e:
                print(f"处理 {fname} 时出错: {e}")

    # 转换为列表格式，以便与原始代码兼容
    for func_name, configs in func_embeddings.items():
        entry = {"func_name": func_name}
        entry.update(configs)
        ebds.append(entry)  # 修正了这里缺少的右括号

    return ebds


class FunctionDataset_Fast(torch.utils.data.Dataset):
    def __init__(self, arr1, arr2):
        self.arr1 = arr1
        self.arr2 = arr2
        assert (len(arr1) == len(arr2))

    def __getitem__(self, idx):
        return self.arr1[idx], self.arr2[idx]

    def __len__(self):
        return len(self.arr1)


def eval(ebds, TYPE1, TYPE2, poolsize, output_file, RECALL_NUM=10):
    """评估使用不同设置编译的同名名之间的相似性"""
    funcarr1 = []
    funcarr2 = []
    func_names = []

    # 查找同时具有两种类型嵌入的同名函数
    for entry in ebds:
        if TYPE1 in entry and TYPE2 in entry:
            funcarr1.append(entry[TYPE1])
            funcarr2.append(entry[TYPE2])
            func_names.append(entry["func_name"])

    pair_num = len(funcarr1)
    print(f'评估 {pair_num} 个函数对')

    if pair_num == 0:
        print(f"未找到 {TYPE1} 和 {TYPE2} 的匹配函数对")
        with open(output_file, 'a') as of:
            of.write(f'{TYPE1} {TYPE2} MRR{poolsize}: 不适用（未找到配对）\n')
            of.write(f'{TYPE1} {TYPE2} Recall@{RECALL_NUM}: 不适用（未找到配对）\n')
        return 0

    ft_valid_dataset = FunctionDataset_Fast(funcarr1, funcarr2)
    dataloader = DataLoader(ft_valid_dataset, batch_size=poolsize, shuffle=False)
    SIMS = []
    Recall_AT_1 = []

    for batch_idx, (anchor, pos) in enumerate(tqdm(dataloader)):
        if torch.cuda.is_available():
            anchor = torch.tensor([item.cpu().detach().numpy() for item in anchor]).cuda()
            pos = torch.tensor([item.cpu().detach().numpy() for item in pos]).cuda()
        else:
            anchor = torch.tensor(anchor)
            pos = torch.tensor(pos)

        batch_size = anchor.shape[0]

        # 计算整个批次的相似性矩阵
        similarity_matrix = F.cosine_similarity(
            anchor.unsqueeze(1),
            pos.unsqueeze(0),
            dim=2
        )

        # 转换为numpy进行处理
        sim_np = similarity_matrix.cpu().numpy()

        for i in range(batch_size):
            # 获取排名（降序）
            rankings = np.argsort(-sim_np[i])

            # 找到正对的位置（应该是i）
            indices = np.where(rankings == i)[0]
            if len(indices) == 0:
                raise ValueError(f"在rankings中未找到位置{i}")
            position = indices[0] + 1  # +1 因为位置从1开始

            # 计算MRR
            SIMS.append(1.0 / position)

            # 计算Recall@N
            Recall_AT_1.append(1 if position <= RECALL_NUM else 0)

    # 计算指标
    mrr = np.mean(SIMS) if SIMS else 0
    recall_at_n = np.mean(Recall_AT_1) if Recall_AT_1 else 0

    print(f"{TYPE1} {TYPE2} MRR{poolsize}: {mrr:.4f}")
    print(f"{TYPE1} {TYPE2} Recall@{RECALL_NUM}: {recall_at_n:.4f}")

    # 写入结果到文件
    with open(output_file, 'a') as of:
        of.write(f'{TYPE1} {TYPE2} MRR{poolsize}: {mrr:.4f}\n')
        of.write(f'{TYPE1} {TYPE2} Recall@{RECALL_NUM}: {recall_at_n:.4f}\n')

    return recall_at_n


if __name__ == '__main__':
    import time

    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
    torch.manual_seed(2023)

    input_dir = r'F:\BinEncoder\dbs\Dataset-1\eval\name_emb'
    output_dir = r'F:\BinEncoder\dbs\Dataset-1\eval\results'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载嵌入
    ebds = load_embeddings(input_dir)

    if not ebds:
        print("未加载任何嵌入。请检查输入目录和文件格式。")
        exit(1)

    # 定义要比较的类型
    RECALL_NUM = 1
    POOLSIZE = 10000

    # 根据文件名格式调整类型定义
    t1 = '_'.join(['clang-7.0', 'arm', '32', 'O0'])
    t2 = '_'.join(['clang-7.0', 'arm', '32', 'O3'])
    t3 = '_'.join(['arm', '32', 'O3'])  # 如果需要更多测试，可以添加更多类型

    output_file = os.path.join(output_dir, f'evaluation_results_{timestamp}.txt')

    # 执行评估
    eval(ebds, t1, t2, poolsize=POOLSIZE, output_file=output_file, RECALL_NUM=RECALL_NUM)
    eval(ebds, t1, t3, poolsize=POOLSIZE, output_file=output_file, RECALL_NUM=RECALL_NUM)
