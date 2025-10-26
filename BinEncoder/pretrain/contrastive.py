import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm

class Config:
    def __init__(self):
        pass

    def training_config(self, batch_size, epochs, lr, weight_decay, device):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.device = device

    def io_config(self, save_path):
        self.save_path = save_path

class TripletDataset(Dataset):
    def __init__(self, anchor_texts, positive_texts, negative_texts, tokenizer, max_length=512):
        self.anchor_texts = anchor_texts
        self.positive_texts = positive_texts
        self.negative_texts = negative_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.anchor_texts)

    def __getitem__(self, idx):
        # 处理单个样本的三元组
        anchor = self._process_text(self.anchor_texts[idx])
        positive = self._process_text(self.positive_texts[idx])
        negative = self._process_text(self.negative_texts[idx])

        # 分词处理（不返回tensor，后续collate统一处理）
        anchor_enc = self.tokenizer(anchor, max_length=self.max_length, truncation=True)
        positive_enc = self.tokenizer(positive, max_length=self.max_length, truncation=True)
        negative_enc = self.tokenizer(negative, max_length=self.max_length, truncation=True)

        return {
            "anchor": anchor_enc,
            "positive": positive_enc,
            "negative": negative_enc
        }

    def _process_text(self, text_dict):
        # 将特征列表转换为字符串
        features = text_dict['features']
        return ', '.join(features).replace('\n', '').strip()

def triplet_collate(batch):
    # 自定义collate函数处理batch数据
    def _pad_encodings(encodings_list):
        input_ids = [torch.tensor(e['input_ids']) for e in encodings_list]
        attention_mask = [torch.tensor(e['attention_mask']) for e in encodings_list]
        return {
            'input_ids': torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True),
            'attention_mask': torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
        }

    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]

    return {
        "anchor": _pad_encodings(anchors),
        "positive": _pad_encodings(positives),
        "negative": _pad_encodings(negatives)
    }

def compute_triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.nn.functional.cosine_similarity(anchor, positive)
    neg_dist = torch.nn.functional.cosine_similarity(anchor, negative)
    losses = torch.clamp(neg_dist - pos_dist + margin, min=0.0)
    return torch.mean(losses)

def train(model, dataloader, config):
    device = torch.device(config.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # 将每个三元组输入移动到设备
            anchor_inputs = {
                'input_ids': batch['anchor']['input_ids'].to(device),
                'attention_mask': batch['anchor']['attention_mask'].to(device)
            }
            positive_inputs = {
                'input_ids': batch['positive']['input_ids'].to(device),
                'attention_mask': batch['positive']['attention_mask'].to(device)
            }
            negative_inputs = {
                'input_ids': batch['negative']['input_ids'].to(device),
                'attention_mask': batch['negative']['attention_mask'].to(device)
            }

            # 获取嵌入
            anchor_emb = model(**anchor_inputs).last_hidden_state[:, 0, :]  # [CLS]嵌入
            positive_emb = model(**positive_inputs).last_hidden_state[:, 0, :]
            negative_emb = model(**negative_inputs).last_hidden_state[:, 0, :]

            # 计算损失
            loss = compute_triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 保存模型
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    model.save_pretrained(config.save_path)
    print(f"Model saved to {config.save_path}")

if __name__ == "__main__":
    # 配置参数
    pretrain_dir = "E:/BinEncoder/pretrain/bert_model/pretraining_bert"
    data_path = "E:/BinEncoder/dbs/Dataset-1/training/extracted_info/triplet/triplets.json"
    save_dir = "E:/BinEncoder/encorder"

    config = Config()
    config.training_config(
        batch_size=8, epochs=30,
        lr=2e-5, weight_decay=0.01,
        device="cuda:0"
    )
    config.io_config(save_dir)

    # 加载数据
    anchor_texts, positive_texts, negative_texts = [], [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            anchor_texts.append(data['anchor'])
            positive_texts.append(data['positive'])
            negative_texts.append(data['negative'])

    # 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained(pretrain_dir)
    model = BertModel.from_pretrained(pretrain_dir)

    # 构建数据集和数据加载器
    dataset = TripletDataset(anchor_texts, positive_texts, negative_texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=triplet_collate,
        drop_last=True
    )

    # 开始训练
    train(model, dataloader, config)

    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
        model.save_pretrained(config.save_path)


