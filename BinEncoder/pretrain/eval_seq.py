import os
import json

def process_json_data(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            processed_data = []
            with open(input_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            for line in lines:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行

                # 处理数据
                for key, value in data.items():
                    if isinstance(value, dict) and 'nverb' in value:
                        nverb_values = value['nverb']
                        formatted_nverb = ', '.join([item[0] for item in nverb_values.values()])
                        processed_data.append({key: formatted_nverb})

            # 写入处理后的数据到输出目录
            with open(output_path, 'w', encoding='utf-8') as file:
                for item in processed_data:
                    json.dump(item, file, ensure_ascii=False)
                    file.write('\n')  # 每个JSON对象占一行

# 使用示例
input_directory = r'F:\BinEncoder\dbs\Dataset-1\eval\seq'
output_directory = r'F:\BinEncoder\dbs\Dataset-1\eval\eval_seq'
process_json_data(input_directory, output_directory)
