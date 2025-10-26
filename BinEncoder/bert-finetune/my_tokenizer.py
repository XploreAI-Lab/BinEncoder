# from transformers import BertTokenizer, BertForMaskedLM
#
modelpath = 'official-model'
# tokenizer = BertTokenizer.from_pretrained(modelpath, use_fast=False)
# model = BertForMaskedLM.from_pretrained(modelpath)
#
# tknz = tokenizer.tokenize('jdxddsnb666')
#
# new_token = ['jxd','jxd666']
# num_added_toks = tokenizer.add_tokens(new_token)
# model.resize_token_embeddings(len(tokenizer))
# print(tknz)
# tokenizer.save_pretrained('mytknz')
#
from transformers import AutoTokenizer, BertForMaskedLM
import torch
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = BertForMaskedLM.from_pretrained(modelpath)
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
