import os
import json
import random
import tqdm

output_file_path = r'E:\BinEncoder\dbs\Dataset-1\training\extracted_info\pair'
output_pos_path = r'E:\BinEncoder\dbs\Dataset-1\training\extracted_info\pair\pos_pairs.json'
output_neg_path = r'E:\BinEncoder\dbs\Dataset-1\training\extracted_info\pair\neg_pairs.json'

def extract_instruction_pairs_from_json(file_path, filename):
    instruction_pairs = []
    name = 'pos_pairs_' + filename
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for func in lines:
            func = json.loads(func)
            addr = list(func.keys())[0]
            g = func[addr]
            nverb = g.get("nverb", [])
            edges = g.get("edges", [])
            for edge in edges:
                start = str(edge[0])
                end = str(edge[1])
                instruction_pairs.append((nverb[start], nverb[end]))
    # with open(os.path.join(output_file_path, name), 'w') as file:
    #     json.dump(instruction_pairs, file, indent=4)
    return instruction_pairs


def extract_instruction_neg_pairs_from_json(file_path, filename):
    instruction_neg_pairs = []
    name = 'neg_pairs_' + filename
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for func in lines:
            func = json.loads(func)
            addr = list(func.keys())[0]
            g = func[addr]
            nverb = g.get("nverb", [])
            k = nverb.keys()
            edges = g.get("edges", [])
            e = []
            for edge in edges:
                e.append([str(edge[0]), str(edge[0])])
            pair_list = []
            while len(pair_list) < len(edges):
                random_ins = random.sample(k, 2)
                if random_ins not in e:
                    pair_list.append((nverb[random_ins[0]], nverb[random_ins[1]]))
                    instruction_neg_pairs.append((nverb[random_ins[0]], nverb[random_ins[1]]))

    # with open(os.path.join(output_file_path, name), 'w') as file:
    #     json.dump(instruction_neg_pairs, file, indent=4)
    return instruction_neg_pairs


def extract_instruction_pairs_from_directory(directory_path):
    all_pos_pairs = []
    all_neg_pairs = []
    for filename in tqdm.tqdm(os.listdir(directory_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            if len(all_pos_pairs) < 200000:
                instruction_pairs = extract_instruction_pairs_from_json(file_path, filename)
                all_pos_pairs.extend(instruction_pairs)
            if len(all_neg_pairs) < 200000:
                instruction_neg_pairs = extract_instruction_neg_pairs_from_json(file_path, filename)
                all_neg_pairs.extend(instruction_neg_pairs)
    print("Positive pairs:", len(all_pos_pairs))
    print("Negative pairs:", len(all_neg_pairs))
    return all_pos_pairs, all_neg_pairs

def save_instruction_pairs_to_json(instruction_pairs, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(instruction_pairs, file, indent=4)


dir_path = r'E:\BinEncoder\dbs\Dataset-1\training\extracted_info\seq'

all_pos_pairs, all_neg_pairs = extract_instruction_pairs_from_directory(dir_path)

save_instruction_pairs_to_json(all_pos_pairs, output_pos_path)
save_instruction_pairs_to_json(all_neg_pairs, output_neg_path)