
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
#sample_submission = pd.read_csv('sample_submission.csv')
CLASSES = list(train['category'].unique())
print(CLASSES)

train['text_length'] = train['text'].apply(len)
test['text_length'] = test['text'].apply(len)

train['occurrence'] = train['text'].map(dict(Counter(train['text'].to_list())))
test['occurrence'] = test['text'].map(dict(Counter(test['text'].to_list())))


# -*- coding: utf-8 -*-
#train = train.drop(train[train['text'] == 'Топовые кроссовки для баскетбола tokenoidtokenoid любимых игроков NBA tokenoidtokenoid 6 лет выполнили 100. 000 заказов'].index)
#train = train.drop(train[train['text'] == 'Премиальный подарок мужчине 33 Шахматы и нарды с именной гравировкой из массива дуба латуни и натуральной кожи. Посмотреть цены'].index)


#test = test.drop(test[test['occurrence'] > 3].index)
#test = test.drop(test[test['text'] == 'СПОЧНО СООБЩЕСТВО ПРОДАЕТСЯ ЗА 1300Р ЗА ПОКУПКОЙ ПИШИТЕ В ЛС 33 ВСЕ ГАРАНТИИ С МЕНЯ 33'].index)

df_train, df_val, df_test = np.split(train.sample(frac=1, random_state=42),
                                     [int(.85*len(train)), int(.95*len(train))])
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
print(len(df_train) + len(df_val) + len(df_test) == len(train))

labels = dict(zip(CLASSES, range(len(CLASSES))))


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


class BertClassifier:
    def __init__(self, model_path, tokenizer_path, data, n_classes=13, epochs=5):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.data = data
        self.device = torch.device('cuda')
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes).cuda()
        self.model = self.model.cuda()

    def preparation(self):
        self.df_train, self.df_val, self.df_test = np.split(self.data.sample(frac=1, random_state=42),
                                                            [int(.85 * len(self.data)), int(.95 * len(self.data))])

        self.train, self.val = CustomDataset(self.df_train, self.tokenizer, phase='train'), CustomDataset(self.df_val,
                                                                                                          self.tokenizer,
                                                                                                          phase='train')
        self.train_dataloader = torch.utils.data.DataLoader(self.train, batch_size=4, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val, batch_size=4)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def fit(self):
        self.model = self.model.train()

        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0
            for train_input, train_label in tqdm(self.train_dataloader):
                train_label = train_label.cuda()
                mask = train_input['attention_mask'].cuda()
                input_id = train_input['input_ids'].squeeze(1).cuda()
                output = self.model(input_id.cuda(), mask.cuda())

                batch_loss = self.loss_fn(output[0], train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output[0].argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            total_acc_val, total_loss_val = self.eval()

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.df_train): .3f} \
            | Train Accuracy: {total_acc_train / len(self.df_train): .3f} \
            | Val Loss: {total_loss_val / len(self.df_val): .3f} \
            | Val Accuracy: {total_acc_val / len(self.df_val): .3f}')

            os.makedirs('checkpoint', exist_ok=True)
            torch.save(self.model, f'checkpoint/BertClassifier{epoch_num}.pt')

        return total_acc_train, total_loss_train

    def eval(self):
        self.model = self.model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in tqdm(self.val_dataloader):
                val_label = val_label.cuda()
                mask = val_input['attention_mask'].cuda()
                input_id = val_input['input_ids'].squeeze(1).cuda()

                output = self.model(input_id.to('cuda'), mask.to('cuda'))

                batch_loss = self.loss_fn(output[0], val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output[0].argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        return total_acc_val, total_loss_val

model_path = 'cointegrated/rubert-tiny'
tokenizer_path = 'cointegrated/rubert-tiny'
bert_tiny = BertClassifier(model_path, tokenizer_path, train, epochs=4)

#bert_tiny.preparation()
#bert_tiny.fit()

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

inference_model = torch.load(f'checkpoint/BertClassifier{bert_tiny.epochs-1}.pt')
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