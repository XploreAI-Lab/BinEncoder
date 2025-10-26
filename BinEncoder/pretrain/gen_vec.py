import json
import torch
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def GetProgramInfo(filename):
    fsplit = filename.split('_')
    compiler = fsplit[1]
    arch = fsplit[2]
    bite = fsplit[3]
    opt = fsplit[4]
    program_name = '{0}-{1}'.format(fsplit[0], fsplit[-1])
    return compiler, arch, bite, opt, program_name

class BertEmbeddingGenerator:
    def __init__(self, model_dir, tokenizer_dir, device='cuda'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def generate_embeddings(self, text):
        # Single processing
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,  # Reduce max_length if possible
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        return embeddings.cpu().numpy()

    def process_json_file(self, json_file_path, output_file_path):
        embeddings = []

        # Read the JSON file line by line
        with open(json_file_path, 'r') as file:
            lines = list(file)
            for line in tqdm(lines, desc=f'Processing {json_file_path}'):  # Process line by line with tqdm
                data = json.loads(line)
                keys = list(data.keys())
                texts = list(data.values())

                for k, text in zip(keys, texts):
                    emb = self.generate_embeddings(text)
                    embeddings.append({'func': k, 'embedding': emb.tolist()})

        # Save embeddings to output JSON file
        with open(output_file_path, 'w') as output_file:
            for entry in embeddings:
                output_file.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    # 设置文件路径和模型目录

    json_input_path = r'F:\BinEncoder\dbs\Dataset-1\eval\eval_seq'  # 输入的 JSON 文件路径
    json_output_path = r'F:\BinEncoder\dbs\Dataset-1\eval\embedding'  # 输出的 JSON 文件路径
    finetuned_model_dir = r'F:\BinEncoder\dbs\Dataset-1\eval\model'  # 训练好的模型的路径
    tokenizer_directory = r'F:\BinEncoder\dbs\Dataset-1\eval\model'  # 分词器的路径

    # 创建嵌入生成器实例并处理输入文件
    generator = BertEmbeddingGenerator(finetuned_model_dir, tokenizer_directory)

    # 遍历json_input_path路径下的所有文件
    for filename in os.listdir(json_input_path):
        if filename.endswith('.json'):
            file_path = os.path.join(json_input_path, filename)
            output_file_path = os.path.join(json_output_path, f'{os.path.splitext(filename)[0]}_embeddings.json')
            generator.process_json_file(file_path, output_file_path)
            print('Embeddings generated and saved to', output_file_path)
