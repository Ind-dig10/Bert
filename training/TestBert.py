import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_cosine_schedule_with_warmup, AdamW
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, phase='test'):
        self.phase = phase

        if self.phase == 'train':
            self.labels = [labels[label] for label in df['category']]
        elif self.phase == 'test':
            self.oid = [oid for oid in df['oid']]

        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        if self.phase == 'train':
            return len(self.labels)
        elif self.phase == 'test':
            return len(self.oid)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_oid(self, idx):
        return np.array(self.oid[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        if self.phase == 'train':
            batch_texts = self.get_batch_texts(idx)
            batch_y = self.get_batch_labels(idx)
            return batch_texts, batch_y
        elif self.phase == 'test':
            batch_texts = self.get_batch_texts(idx)
            batch_oid = self.get_batch_oid(idx)
            return batch_texts, batch_oid

test_dataset = CustomDataset(test, bert_tiny.tokenizer, phase='test')
test_dataloader = DataLoader(test_dataset, batch_size=4)


def inference(model, dataloader):
    all_oid = []
    all_labels = []
    label_prob = []

    model.cuda()
    model.eval()
    with torch.no_grad():
        for test_input, test_oid in tqdm(dataloader):
            test_oid = test_oid.cuda()
            mask = test_input['attention_mask'].cuda()
            input_id = test_input['input_ids'].squeeze(1).cuda()
            output = model(input_id, mask)
            all_oid.extend(test_oid)
            all_labels.extend(torch.argmax(output[0].softmax(1), dim=1))

            for prob in output[0].softmax(1):
                label_prob.append(prob)
        return ([oid.item() for oid in all_oid], [CLASSES[labels] for labels in all_labels], label_prob)

inference_model = torch.load(f'/checkpoint/BertClassifier{bert_tiny.epochs-1}.pt')
inference_result = inference(inference_model, test_dataloader)

oid = [i for i in inference_result[0]]
labels = [i for i in inference_result[1]]
prob = [i for i in inference_result[2]]
print(len(dict(zip(oid, labels))))
print(len(set(oid) & set(test['oid'].unique())))
print(len(set(oid) & set(test['oid'].unique())))

detached_prob = []
for i in prob:
    detached_prob.append(i.cpu().numpy())

data = {'oid': oid, 'category': labels, 'probs': detached_prob}
submit = pd.DataFrame(data)
submit['label_int'] = submit['category'].apply(lambda x: CLASSES.index(x))

label_int = submit['label_int'].to_list()
probs = submit['probs'].to_list()
res = []
for indx, tensor in enumerate(probs):
    res.append(tensor[label_int[indx]])
submit['prob'] = res
del submit['probs'], submit['label_int']
tmp_submit = pd.DataFrame(submit.groupby(by=['oid', 'category']).sum().reset_index())

oid = tmp_submit['oid'].to_list()
category = tmp_submit['category'].to_list()
prob = tmp_submit['prob'].to_list()

res = {}
for indx, id in enumerate(oid):
    if id not in res:
        res[id] = (category[indx], prob[indx])

submit_data = {k: v[0] for k, v in res.items()}
oid = list(submit_data.keys())
category = list(submit_data.values())
pd.DataFrame({'oid': oid, 'category': category}).to_csv('submission.csv', index=False)