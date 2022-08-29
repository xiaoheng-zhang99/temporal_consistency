# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel
from torch import nn
import torch
import torch.nn.functional as F


# block for speech encoder (500,768)
class LSTM_speech(nn.Module):
    def __init__(self):
        super(LSTM_speech, self).__init__()
        self.LSTM = nn.LSTM(768, 6, 3)  # 特征数量；隐藏层神经元数；隐藏层数
        self.forward1 = nn.Linear(6, 768)

    def forward(self, x):
        x, _ = self.LSTM(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.forward1(x)
        # x=x.view(s,b,-1)
        return x


# block for text encoder (150,768)
class CNNBert(nn.Module):
    def __init__(self):
        super(CNNBert, self).__init__()
        filter_sizes = [1, 3, 5]
        num_filter = 1
        pretrained_model = "C:/Users/s123c/Desktop/bert-base-chinese/"
        self.bert_model = BertModel.from_pretrained(pretrained_model, output_hidden_states=True)
        self.convs1 = nn.ModuleList([nn.Conv1d(768, 1, (768, K)) for K in filter_sizes])  # 待修改

    def forward(self, x, input_masks, token_type_ids):
        x = self.bert_model(x, attention_mask=input_masks, token_type_ids=token_type_ids)[2][-4:]
        x = torch.stack(x, dim=1)
        # bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 获得预训练模型的输出
        # bert_cls_hidden_state = bert_output[1]
        # 将768维的向量输入到线性层映射为二维向量
        x = [conv(x) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x


pretrained_model = "C:/Users/s123c/Desktop/bert-base-chinese/"
bert_model = BertModel.from_pretrained(pretrained_model, output_hidden_states=True)



# block after fusion

class fusion_CNN(nn.Module):
    def __init__(self, embed_size, max_features):
        super(fusion_CNN, self).__init__()

        filter_sizes = [1, 3, 5]
        num_filters = 1
        self.hidden_size = 128
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=False, batch_first=True)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 4)

    def forward(self, x):
        x = self.lstm(x)
        x = x.convs1(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = torch.cat(x, 1)
        logit = self.fc1(x)
        x = F.softmax(logit)
        return logit
'''
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# 输入数据
a = np.zeros((4, 768))
b = np.zeros((4, 768))
for i in range(768):
    for k in range(4):
        a[k, i] = k + np.random.rand(1)
        b[k, i] = k
#print(a,b)
# 给数据加同一个噪声然后进行预测


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(768, 6, 3) # 第一个参数是特征系数（也就是需要预测的特征数量）输入的变量的二进制长度，
    # 第二个变量是隐藏层的神经元个数，第三个是隐藏层的个数
        self.forward1 = nn.Linear(6,768)

    def forward(self, x):
        x, _ = self.lstm(x)
        s, b, h = x.shape
    #print(s,b,h)
        x = x.view(s*b, h)
        x = self.forward1(x)
        x = x.view(s, b, -1)
        return x


train_date = np.reshape(a, (4, 1, -1))
print(train_date.shape)
train_y_date = np.reshape(b, (4, 1, -1))
#train_date = train_date.reshape(4, 1, 81)
train_date_tensor = torch.from_numpy(train_date).to(torch.float32)
#train_y_date = train_y_date.reshape(4, 1, 81)
train_date_y_tensor = torch.from_numpy(train_y_date).to(torch.float32)
lstm_model = CNNBert()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

for i in range(800):
    out_put = lstm_model(train_date_tensor)
    loss = loss_function(out_put, train_date_y_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (loss.item() < 1e-4):
    print('successful',i)
'''