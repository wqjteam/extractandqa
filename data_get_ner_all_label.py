# -*- coding: utf-8 -*-
import os
import sys
from collections import OrderedDict
from functools import partial

import pandas as pd
from sklearn import metrics
import numpy as np
import torch
import torch.utils.data as Data
import torchmetrics
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorForTokenClassification, get_cosine_schedule_with_warmup, AutoConfig
from visdom import Visdom

from PraticeOfTransformers import Utils
from PraticeOfTransformers.CustomModelForNer import BertForNerAppendBiLstmAndCrf

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定GPU编号 多gpu训练
print('--------------------sys.argv:%s-------------------' % (','.join(sys.argv)))

model_name = 'bert-base-chinese'
batch_size = 2
epoch_size = 500
learning_rate = 1e-5
weight_decay = 0.01  # 最终目的是防止过拟合
full_fine_tuning = True

if len(sys.argv) >= 2:
    batch_size = int(sys.argv[1])
if len(sys.argv) >= 3:
    epoch_size = int(sys.argv[2])

ner_id_label = {0: '[CLS]', 1: '[SEP]', 2: 'O', 3: 'B-ORG', 4: 'B-PER', 5: 'B-LOC', 6: 'B-TIME', 7: 'B-BOOK',
                8: 'I-ORG', 9: 'I-PER', 10: 'I-LOC', 11: 'I-TIME', 12: 'I-BOOK'}
ner_label_id = {}
for key in ner_id_label:
    ner_label_id[ner_id_label[key]] = key

id2labelids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
id2labelvalues = ['[CLS]', '[SEP]', 'O', 'B-ORG', 'B-PER', 'B-LOC', 'B-TIME', 'B-BOOK', 'I-ORG', 'I-PER', 'I-LOC',
                  'I-TIME', 'I-BOOK']

key_nums = {
    'B-ORG': 0, 'B-PER': 0, 'B-LOC': 0, 'B-TIME': 0, 'B-BOOK': 0
}

# 加载数据集
nerdataset = Utils.convert_ner_data('data/origin/intercontest/relic_ner_handlewell.json')
# nerdataset = list(filter(lambda x: ''.join(x[0]).startswith("小双桥遗址"), nerdataset))
# nerdataset = nerdataset[0:100]
for data in nerdataset:
    for label in data[1]:
        if label.startswith('B'):
            key_nums[label] = key_nums[label] + 1

print(key_nums)

data = pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)

special_nums = 0

for i_item, j_item in data.iterrows():
    if len(j_item['keyword']) > 1:
        print(j_item['keyword'])
    else:
        special_nums = special_nums + 1

print(special_nums)