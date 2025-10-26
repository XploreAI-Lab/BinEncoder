import os
import json
import glob
import copy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertTokenizerFast
import time

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
        """
        :param mlm_probability: 被mask的token总数
        :param special_token_mask: 特殊token
        :param prob_replace_mask: 被替换成[MASK]的token比率
        :param prob_replace_rand: 被随机替换成其他token比率
        :param prob_keep_ori: 保留原token的比率
        """
        assert sum([prob_replace_mask, prob_replace_rand, prob_keep_ori]) == 1, ValueError(
            "Sum of the probs must equal to 1.")
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori

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
            from_path,
            save_path,
    ):
        self.from_path = from_path
        self.save_path = save_path

class TrainDataset(Dataset):
    """
    注意：由于没有使用data_collator，batch放在dataset里边做，
    因而在dataloader出来的结果会多套一层batch维度，传入模型时注意squeeze掉
    """

    def __init__(self, input_texts, tokenizer, config):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.config = config
        self.ori_inputs = copy.deepcopy(input_texts)

    def __len__(self):
        return len(self.input_texts) // self.config.batch_size

    def __getitem__(self, idx):
        batch_text = self.input_texts[: self.config.batch_size]
        features = self.tokenizer(batch_text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        inputs, labels = self.mask_tokens(features['input_ids'])
        batch = {"inputs": inputs, "labels": labels}
        self.input_texts = self.input_texts[self.config.batch_size:]
        if not len(self):
            self.input_texts = self.ori_inputs

        return batch

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
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
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # in labels , special tokens and 0.85 of the normal tokens --> -100  keep the 15% of the normal tokens unchanged as the mask token.

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.config.prob_replace_rand / (1 - self.config.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def plot_loss(name, loss):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    # par1 = host.twinx()  # 共享x轴

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel(name + "-loss")
    # par1.set_ylabel(name + "-accuracy")

    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    # p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()
    plt.show()

def train(model, train_dataloader, config):
    """
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    """
    assert config.device.startswith('cuda') or config.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(config.device)

    model.to(device)

    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")
    # batch_num = len(train_dataloader/config.batch_size)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    # train_acc_list = []
    # train_loss_list = []
    for cur_epc in range(int(config.epochs)):
        training_loss = 0.0
        print("Epoch: {}".format(cur_epc + 1))
        model.train()

        for batch in tqdm(train_dataloader, desc='Step'):
            input_ids = batch['inputs'].squeeze(0).to(device)
            labels = batch['labels'].squeeze(0).to(device)
            loss = model(input_ids=input_ids, labels=labels).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            training_loss += loss.item()
            if not loss.item :
                print(labels)

        avg_loss = training_loss/len(train_dataloader)
        print("Training avg loss: ", avg_loss)
        # train_loss_list.append(avg_loss)
    # plot_loss('mlm_train',train_loss_list)



data_loading_dir = 'data/labeled_cls_text/libgmp-10.4.1'
data_paths = glob.glob(data_loading_dir + os.sep + '*')

for cur_path in data_paths:

    # data_loading_path = './data/labeled_cls_text/busybox/busybox_1.27.2_arm_O0_features_sample20000block(2023-02-24-14-47-13).csv'
    timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
    cur_file = '_'.join(cur_path.split(os.sep)[-1].split('_')[:4])
    print('cur_file:',cur_file)

    model_dir = './data/model_save' + os.sep + data_loading_dir.split(os.sep)[-1]
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_save_path = model_dir + os.sep + cur_file + timestamp

    config = Config()
    config.mlm_config()
    config.training_config(batch_size=16, epochs=12, learning_rate=1e-5, weight_decay=0, device='cuda:0')
    config.io_config(from_path=cur_path,
                     save_path=model_save_path)

    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    bert_mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    with open(config.from_path,'r') as fp :
        training_texts =fp.readlines()

    train_dataset = TrainDataset(training_texts, bert_tokenizer, config)
    train_dataloader = DataLoader(train_dataset,shuffle=True)

    train(model=bert_mlm_model, train_dataloader=train_dataloader, config=config)
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    bert_mlm_model.save_pretrained(config.save_path)

