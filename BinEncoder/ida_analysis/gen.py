import json
import numpy as np


def generate_random_embeddings(num_embeddings, embedding_dim):
    embeddings = []
    for _ in range(num_embeddings):
        embedding = np.random.rand(embedding_dim).tolist()
        entry = {"embedding": [embedding]}
        embeddings.append(entry)
    return embeddings


def save_embeddings_to_json(embeddings, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in embeddings:
            json.dump(entry, f)
            f.write('\n')


if __name__ == "__main__":
    num_embeddings = 62605
    embedding_dim = 64  # 根据您提供的嵌入格式，设定为64维
    output_file_path = r'F:\BinEncoder\dbs\Dataset-1\eval\pool_embeddings.json'

    embeddings = generate_random_embeddings(num_embeddings, embedding_dim)
    save_embeddings_to_json(embeddings, output_file_path)
    print(f"已生成包含 {num_embeddings} 个嵌入的 JSON 文件，保存路径为 {output_file_path}")