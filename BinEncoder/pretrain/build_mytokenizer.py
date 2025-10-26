from transformers import BertForMaskedLM, BertTokenizer
import os

if __name__ == '__main__':
    add_vocal = r"E:\BinEncoder\dbs\Dataset-1\training\extracted_token\add_vocal.txt"
    read_dir = r"E:\BinEncoder\pretrain\bert_model\pretraining_bert"
    save_dir = r"E:\BinEncoder\pretrain\bert_model\pretraining_bert"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(add_vocal,'r') as fp:
        read_tokens = fp.readlines()
        add_tokens = []
        for i in read_tokens:
            add_tokens.append(i.strip())
        tokenizer = BertTokenizer.from_pretrained(read_dir)
        model = BertForMaskedLM.from_pretrained(read_dir)
        tokenizer.add_tokens(add_tokens)

        model.resize_token_embeddings(len(tokenizer))

        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)