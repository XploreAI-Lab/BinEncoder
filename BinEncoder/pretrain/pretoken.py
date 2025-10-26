import json
import os

def extract_nverb_edge(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    function_info = {}

    # 遍历 JSON 层次结构
    for key, value in data.items():
        for inner_key, inner_value in value.items():
            if 'ISCG' in inner_value:
                # 初始化 inner_key 的字典
                if inner_key not in function_info:
                    function_info[inner_key] = {"nverb": {}, "edges": []}

                # 提取 nverbs
                if 'nverbs' in inner_value['ISCG']:
                    function_info[inner_key]['nverb'] = inner_value['ISCG']['nverbs']
                # 提取 edges
                if 'edges' in inner_value['ISCG']:
                    function_info[inner_key]['edges'] = inner_value['ISCG']['edges']

    return function_info

def extract_from_all_json_files(directory, out_path):
    all_function_info = {}
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory, filename)
            function_info = extract_nverb_edge(json_file_path)
            all_function_info[filename] = function_info

            # 为每个文件生成输出路径
            output_file_path = os.path.join(out_path, f"{os.path.splitext(filename)[0]}_extracted_info.json")
            output_data = {
                "function Information": function_info,
            }
            save_to_json_file(output_data, output_file_path)

            print(f"提取的信息已经保存到 {output_file_path}")

    return all_function_info

def save_to_json_file(data, file_path):
    """将数据保存到 JSON 文件"""
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # 添加换行符


if __name__ == '__main__':

    directory_path = r"F:\BinEncoder\dbs\Dataset-1\eval\feature"
    out_path = r"F:\BinEncoder\dbs\Dataset-1\eval\info"

    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)

    # 提取信息并保存
    extract_from_all_json_files(directory_path, out_path)