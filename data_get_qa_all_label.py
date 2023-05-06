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

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]



# passage_keyword_json = passage_keyword_json[passage_keyword_json.nsp == 1]
# passage_keyword_json = passage_keyword_json[passage_keyword_json['sentence'].apply(lambda x: '长治市博物馆，' in x)]

passage_keyword_json = passage_keyword_json.explode("q_a").values

sent = ['我爱北京天安门，天安门上太阳升', '我爱北京中南海，毛主席在中南还', '改革开放好，我哎深圳，深圳是改革开放先驱']
question = ['我爱什么?', '毛主席在哪?', '谁是改革开放先驱']



# 创建一个实例，参数是tokenizer，如果不是batch的化，就采用tokenizer.encode_plus

# print(encoded_dict)

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}


# print(output)


def create_batch(data):
    text, question_answer, keyword, nsp = zip(*data)  # arrat的四列 转为tuple

    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]
    returnpassages=[]
    returnquestion=[]
    returnanswer = []
    nsps = list(nsp)  # tuple 转为list

    keywords = [kw[0] for kw in keyword]  # tuple 转为list 变成了双重的list 还是遍历转
    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置 需要在进行encode之后再进行添加
    end_positions_labels = []  # 记录终止始位置 需要在进行encode之后再进行添加

    for array_index, textstr in enumerate(list(text)):
        question=questions[array_index]
        answer=answers[array_index]
        start_in = CommonUtil.get_first_index_in_array(textstr, answer)  # 这方法在data_collator存在，不再重复写了
        if start_in != -1 and nsps[array_index] == 1:  # 判断是否存在
            returnpassages.append(textstr)
            returnquestion.append(question)
            nsp_labels.append(nsp_label_id.get(True))
            returnanswer.append({"answer": answer,"start":start_in,"end":start_in + len(answer)})


            print(textstr[start_in:start_in + len(answer)])
        else:
            pass
            # nsp_labels.append(nsp_label_id.get(False))
            #
            # sep_in = textstr.index(0)
            # start_positions_labels.append(sep_in)
            # end_positions_labels.append(sep_in)
    return zip(returnpassages,returnquestion,returnanswer)



# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
q_a=create_batch(passage_keyword_json)
for qa in q_a:
    print(qa)
