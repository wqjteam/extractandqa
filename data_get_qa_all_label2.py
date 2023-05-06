# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torchmetrics
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam
from visdom import Visdom
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, get_cosine_schedule_with_warmup

import CommonUtil
from PraticeOfTransformers import Utils
from PraticeOfTransformers.DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial
from PraticeOfTransformers.CustomModelForNSPQA import BertForUnionNspAndQA, NspAndQAConfig
from PraticeOfTransformers.DataCollatorForWholeWordMaskOriginal import DataCollatorForWholeWordMaskOriginal
from PraticeOfTransformers.DataCollatorForWholeWordMaskSpecial import DataCollatorForWholeWordMaskSpecial
import sys



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定GPU编号 多gpu训练
batch_size = 2
epoch_size = 1000
learning_rate = 1e-5
weight_decay = 0.01  # 最终目的是防止过拟合
full_fine_tuning = True

print('--------------------sys.argv:%s-------------------' % (','.join(sys.argv)))
if len(sys.argv) >= 2:
    batch_size = int(sys.argv[1])
if len(sys.argv) >= 3:
    epoch_size = int(sys.argv[2])



'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)
# passage_keyword_json['q_a'] 和 passage_keyword_json['q_a'].q_a 一样
passage_keyword_json = passage_keyword_json[passage_keyword_json['nsp'].apply(lambda x:  x==1)]

# passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) > 10)]

print(passage_keyword_json.head(1000))