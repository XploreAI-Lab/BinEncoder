import os
import json
import glob
import copy

import numpy
from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForPreTraining, BertTokenizerFast
import time
import numpy as np


class Config:
    def __init__(self):
        pass

    def mlm_config(
            self,
            mlm_probability=0.15,
            special_tokens_mask=None,
            prob_replace_mask=0.8,
            prob_replace_rand=0.1,
            prob_keep_ori=0.1,
    ):
        assert sum([prob_replace_mask, prob_replace_rand, prob_keep_ori]) == 1, ValueError(
            "Sum of the probs must equal to 1.")
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori

    def nsp_config(
            self,
            nsp_probability=0.5,
    ):
        self.nsp_probability = nsp_probability

    def training_config(
            self,
            batch_size,
            epochs,
            learning_rate,
            weight_decay,
            device,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

    def io_config(
            self,
            mlm_from_path,
            nsp_from_path,
            save_path,
    ):
        self.mlm_from_path = mlm_from_path
        self.nsp_from_path = nsp_from_path
        self.save_path = save_path


class TrainDataset(Dataset):

    def __init__(self, pairs, labels, tokenizer, config):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.config = config
        self.ori_pairs = copy.deepcopy(pairs)
        self.ori_labels = copy.deepcopy(labels)
        self.labels = labels

    def __len__(self):
        return len(self.pairs) // self.config.batch_size

    def __getitem__(self, idx):
        if len(self.pairs) < self.config.batch_size:
            print("\nResetting dataset state...")
            self.pairs = self.ori_pairs
            self.labels = self.ori_labels

        batch_nsp_pairs = self.pairs[: self.config.batch_size]
        batch_nsp_labels = self.labels[: self.config.batch_size]

        nor_batch_nsp_text = list()
        for insts in batch_nsp_pairs:
            insts = str(insts).replace('\n', '')
            nor_batch_nsp_text.append(insts)

        nsp_features = self.tokenizer(nor_batch_nsp_text, max_length=512, truncation=True, padding='max_length',
                                      return_tensors='pt')

        mlm_inputs, mlm_labels = self.mask_tokens(nsp_features['input_ids'])

        batch = {
            "input_ids": mlm_inputs,
            "token_type_ids": nsp_features['token_type_ids'],
            "attention_mask": nsp_features['attention_mask'],
            "mlm_labels": mlm_labels,
            "nsp_labels": torch.tensor(batch_nsp_labels)
        }

        self.pairs = self.pairs[self.config.batch_size:]
        self.labels = self.labels[self.config.batch_size:]

        return batch

    def create_nsp_inputs(self, batch_text):
        combined_inputs = []
        labels = []

        for i in range(0, len(batch_text), 2):
            if i + 1 < len(batch_text) and np.random.rand() < self.config.nsp_probability:
                combined_inputs.append(batch_text[i] + " " + self.tokenizer.sep_token + " " + batch_text[i + 1])
                labels.append(0)
            else:
                random_index = np.random.randint(len(batch_text))
                combined_inputs.append(batch_text[i] + " " + self.tokenizer.sep_token + " " + batch_text[random_index])
                labels.append(1)

        return combined_inputs, labels

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)
        if self.config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.config.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


def train(model, train_dataloader, config):
    assert config.device.startswith('cuda') or config.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(config.device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    for cur_epc in range(int(config.epochs)):
        training_loss = 0.0
        print(f"\n--- Epoch: {cur_epc + 1} ---")
        model.train()

        for batch in tqdm(train_dataloader, desc='Step', total=len(train_dataloader)):

            input_ids = batch['input_ids'].squeeze(0).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(0).to(device)
            attention_mask = batch['attention_mask'].squeeze(0).to(device)
            mlm_labels = batch['mlm_labels'].squeeze(0).to(device)
            nsp_labels = batch['nsp_labels'].squeeze(0).to(device)

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=mlm_labels,
                next_sentence_label=nsp_labels
            )
            loss = outputs.loss

            if isinstance(loss, torch.Tensor):
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            training_loss += loss.item()

        avg_loss = training_loss / len(train_dataloader)
        print(f"Epoch {cur_epc + 1} Training avg loss: {avg_loss:.5f}")


if __name__ == '__main__':
    pretrain_dir = r"E:\BinEncoder\pretrain\bert_model\pretraining_bert"
    pos_pairs_dir = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info\pair\pos_pairs.json"
    neg_pairs_dir = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_info\pair\neg_pairs.json"

    model_dir_name = "pretrain_run"
    model_dir = r"E:\BinEncoder\pretrain\bert_model" + os.sep + model_dir_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
    model_save_path = model_dir + os.sep + 'combined_model_' + timestamp

    config = Config()
    config.mlm_config()
    config.nsp_config()
    config.training_config(batch_size=8, epochs=30, learning_rate=1e-5, weight_decay=0, device='cuda:0')
    config.io_config(mlm_from_path=None, nsp_from_path=[pos_pairs_dir, neg_pairs_dir], save_path=model_save_path)

    bert_tokenizer = BertTokenizerFast.from_pretrained(pretrain_dir)
    bert_model = BertForPreTraining.from_pretrained(pretrain_dir)

    pairs = []
    labels = []

    print(f"Loading positive pairs from: {pos_pairs_dir}")
    with open(pos_pairs_dir, 'r', encoding='utf-8') as fp:
        pos_pairs = json.load(fp)
        pairs.extend(pos_pairs)
        labels.extend([0] * len(pos_pairs))

    print(f"Loading negative pairs from: {neg_pairs_dir}")
    with open(neg_pairs_dir, 'r', encoding='utf-8') as fp:
        neg_pairs = json.load(fp)
        pairs.extend(neg_pairs)
        labels.extend([1] * len(neg_pairs))

    print(f"Loaded {len(pairs)} total pairs ({len(pos_pairs)} pos, {len(neg_pairs)} neg).")

    print("Shuffling data...")
    temp = list(zip(pairs, labels))
    np.random.shuffle(temp)
    pairs, labels = zip(*temp)
    pairs, labels = list(pairs), list(labels)
    print("Shuffling complete.")

    train_dataset = TrainDataset(pairs, labels, bert_tokenizer, config)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda x: x[0]
    )

    train(model=bert_model, train_dataloader=train_dataloader, config=config)

    print(f"Training complete. Saving model to: {config.save_path}")
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    bert_model.save_pretrained(config.save_path)
    bert_tokenizer.save_pretrained(config.save_path)
    print("Model saved.")