import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def generate_tsne_plot(json_file_path, output_image_path, highlight_index=0):
    # 从 JSON 文件读取数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 提取函数嵌入
    embeddings = []
    for entry in data:
        # 确保嵌入数据被正确展平为一维
        embedding = np.array(entry['embedding']).flatten()
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # 使用 T-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制 T-SNE 图，设置颜色为浅蓝色
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='#87CEFA')

    # 使用红色五角星标记指定的嵌入
    highlight_point = plt.scatter(embeddings_2d[highlight_index, 0], embeddings_2d[highlight_index, 1],
                                  color='red', marker='*', s=200)  # s 是标记的大小

    # 添加固定内容的注释框
    fixed_annotation = "Highlighted Function"
    plt.annotate(fixed_annotation,
                 xy=(embeddings_2d[highlight_index, 0], embeddings_2d[highlight_index, 1]),
                 xytext=(10, 10), textcoords='offset points',
                 color='black', fontsize=12,  # 修改字体颜色为黑色
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))

    # 去掉坐标轴上的数字，只保留黑色边框
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_color('black')

    # 去掉图的标题和横纵坐标轴标签
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')

    # 保存图片到指定位置
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    # json_file_path = r'F:\BinEncoder\dbs\Dataset-1\eval\name_emb\binutils-2.30_clang-7.0_arm_32_O3_c++filt.elf_acfg_disasm_extracted_info_embeddings.json'
    json_file_path = r'F:\BinEncoder\dbs\Dataset-1\eval\pool_embeddings.json'
    output_image_path = r'C:\Users\tianh\Desktop\tsne_visualization.png'
    generate_tsne_plot(json_file_path, output_image_path, highlight_index=0)
