import json
import numpy as np


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def find_most_similar_function(json_file_path, target_func_name):
    """
    在JSON文件中找到与目标函数余弦相似度最高的函数
    :param json_file_path: JSON文件路径
    :param target_func_name: 目标函数名称
    :return: 最相似的函数名称和对应的余弦相似度
    """
    target_embedding = None
    all_embeddings = []
    all_func_names = []

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            func_name = data['func_name']
            embedding = np.array(data['embedding']).flatten()
            all_embeddings.append(embedding)
            all_func_names.append(func_name)
            if func_name == target_func_name:
                target_embedding = embedding

    if target_embedding is None:
        raise ValueError(f"目标函数 {target_func_name} 未在JSON文件中找到")

    max_similarity = -1
    most_similar_func = None
    for i, embedding in enumerate(all_embeddings):
        similarity = cosine_similarity(target_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_func = all_func_names[i]

    return most_similar_func, max_similarity


if __name__ == "__main__":
    json_file_path = 'your_file.json'  # 替换为实际的JSON文件路径
    target_func_name = 'string_prepend_0'  # 替换为你要查找的目标函数名称
    most_similar_func, similarity = find_most_similar_function(json_file_path, target_func_name)
    print(f"与 {target_func_name} 余弦相似度最高的函数是 {most_similar_func}，相似度为 {similarity}")
