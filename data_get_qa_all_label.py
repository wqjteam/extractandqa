# -*- coding: utf-8 -*-
import math
import os
import random
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


'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)
# passage_keyword_json['q_a'] 和 passage_keyword_json['q_a'].q_a 一样
passage_keyword_json = passage_keyword_json[passage_keyword_json['nsp'].apply(lambda x: x == 1)]

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

# passage_keyword_json = passage_keyword_json[passage_keyword_json.nsp == 1]
# passage_keyword_json = passage_keyword_json[passage_keyword_json['sentence'].apply(lambda x: '长治市博物馆，' in x)]
passage_keyword_json['sequence_index'] = passage_keyword_json.index
passage_keyword_json = passage_keyword_json.explode("q_a").values

# 创建一个实例，参数是tokenizer，如果不是batch的化，就采用tokenizer.encode_plus

# print(encoded_dict)

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}


# print(output)


def create_batch(data):
    text, question_answer, keyword, nsp, seq = zip(*data)  # arrat的四列 转为tuple

    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]
    nsps = list(nsp)  # tuple 转为list
    seq = list(seq)
    returnpassages = []
    returnquestion = []
    returnanswer = []
    returnseq = []
    returntuple = []

    keywords = [kw[0] for kw in keyword]  # tuple 转为list 变成了双重的list 还是遍历转
    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置 需要在进行encode之后再进行添加
    end_positions_labels = []  # 记录终止始位置 需要在进行encode之后再进行添加

    for array_index, textstr in enumerate(list(text)):
        question = questions[array_index]
        answer = answers[array_index]
        start_in = CommonUtil.get_first_index_in_array(textstr, answer)  # 这方法在data_collator存在，不再重复写了
        seqindex = seq[array_index]
        if start_in != -1 and nsps[array_index] == 1:  # 判断是否存在
            returnpassages.append(textstr)
            returnquestion.append(question)
            nsp_labels.append(nsp_label_id.get(True))
            answertuple = {"answer": answer, "start": start_in, "end": start_in + len(answer)}
            returnanswer.append(answertuple)
            returnseq.append(seqindex)
            returntuple.append((textstr, question, answertuple))

            # print(textstr[start_in:start_in + len(answer)])
        else:
            pass
            # nsp_labels.append(nsp_label_id.get(False))
            #
            # sep_in = textstr.index(0)
            # start_positions_labels.append(sep_in)
            # end_positions_labels.append(sep_in)

    return pd.DataFrame({'seq': returnseq, 'data': returntuple})


# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
q_a = create_batch(passage_keyword_json)


def collect_list_passage(group):
    passage = ''
    q_a = []
    for data in group:
        passage = data[0]
        answer = data[2]
        q_a.append({'question': data[1],
                    'answer': {'text': answer.get('answer'), 'start': answer.get('start'), 'end': answer.get('end')}})
        # print(data)
    return passage, q_a


def collect_list_qa(group):
    q_a = []
    for data in group:
        answer = data[2]
        q_a.append({'question': data[1],
                    'answer': {'text': answer.get('answer'), 'start': answer.get('start'), 'end': answer.get('end')}})
        # print(data)
    return q_a


q_a = q_a.groupby('seq').agg({'data': collect_list_passage})
q_a['passage'] = q_a['data'].map(lambda x: x[0])
q_a['q_a'] = q_a['data'].map(lambda x: x[1])
q_a = q_a.drop('data', axis=1)

cmrcdata = pd.read_json('./data/origin/cmrc/cmrc2018_train.json')


def getqa_answer(data):
    data = data['data']
    return data.get('paragraphs')[0].get('context'), data.get('paragraphs')[0].get('qas')


def orgnize_answer(datas):
    q_a = []
    for data in datas:
        answer = data.get('answers')[0]
        q_a.append({'question': data.get('question'),
                    'answer': {'text': answer.get('text'), 'start': answer.get('answer_start'),
                               'end': answer.get('answer_start') + len(answer.get('text'))}})
    return q_a


cmrcdata[['passage', 'q_a']] = cmrcdata.apply(getqa_answer, axis=1, result_type='expand')
cmrcdata['q_a'] = cmrcdata['q_a'].map(orgnize_answer)
cmrcdata = cmrcdata.drop(['version', 'data'], axis=1)

union_qa_data = pd.concat([q_a, cmrcdata], axis=0)
# union_qa_data.to_json('data/origin/intercontest/union_qa_positive_negate.json', force_ascii=False, orient='records',
#                       lines=True)
default_df = union_qa_data.copy(deep=True)
default_df['nsp'] = default_df['q_a'].map(lambda x: 1)
union_qa_data = union_qa_data.reset_index()
index_size = union_qa_data.index.size
union_qa_data['new_temp_index'] = union_qa_data.index


def match_error_multiple(sentence, index_size,q_a_name):
    # 获取需要去除的index
    current_index = sentence['new_temp_index']
    # 生成所有备选index，移除现在的index，然后在其中随机选择

    alternativearray = np.arange(0, index_size).tolist()
    alternativearray.remove(current_index)
    randomindex = random.randrange(len(alternativearray))
    q_a = union_qa_data.iloc[randomindex][q_a_name]
    return q_a, 0  # 0的话为false


union_qa_data[['q_a', 'nsp']] = union_qa_data.apply(lambda row: match_error_multiple(row, index_size,'q_a'), axis=1,
                                                    result_type='expand')

union_qa_error_postivate = pd.concat([default_df, union_qa_data.drop(['index', 'new_temp_index'], axis=1)], axis=0)

# union_qa_error_postivate = union_qa_error_postivate.sample(frac=1)  # 乱序处理
# union_qa_error_postivate.to_json('data/origin/intercontest/union_culture_kiwi_qa_error_postivate_train.json', force_ascii=False,orient='records', lines=True)
# passage_keyword_json = pd.read_json("./data/origin/intercontest/union_culture_kiwi_qa_error_postivate.json",
#                                     orient='records',
#                                     lines=True).head(100)


def get_organize_data_bywiki(filepath):
    cmrcdata = pd.read_json(filepath)

    cmrcdata[['passage', 'q_a']] = cmrcdata.apply(getqa_answer, axis=1, result_type='expand')
    cmrcdata['q_a'] = cmrcdata['q_a'].map(orgnize_answer)
    cmrcdata = cmrcdata.drop(['version', 'data'], axis=1)
    default_df = cmrcdata.copy(deep=True)
    default_df['nsp'] = default_df['q_a'].map(lambda x: 1)
    cmrcdata = cmrcdata.reset_index()
    cmrcdata['new_temp_index'] = cmrcdata.index
    index_size = cmrcdata.index.size
    cmrcdata[['q_a', 'nsp']] = cmrcdata.apply(lambda row: match_error_multiple(row, index_size,'q_a'), axis=1,
                                                        result_type='expand')

    union_qa_error_postivate = pd.concat([default_df, cmrcdata.drop(['index', 'new_temp_index'], axis=1)], axis=0)

    union_qa_error_postivate = union_qa_error_postivate.sample(frac=1)  # 乱序处理
    return union_qa_error_postivate

dev_data=get_organize_data_bywiki('./data/origin/cmrc/cmrc2018_dev.json')
test_data=get_organize_data_bywiki('./data/origin/cmrc/cmrc2018_trial.json')
all_data=pd.concat([union_qa_error_postivate, dev_data,test_data], axis=0).sample(frac=1)

all_data = all_data.reset_index()


alldatasize=len(all_data)
train_size=math.ceil(alldatasize*0.8)
dev_size=math.ceil(alldatasize*0.1)
test_size=alldatasize-train_size-dev_size

train_data=all_data.iloc[:train_size,:]
dev_data=all_data.iloc[train_size:train_size+dev_size,:]
test_data=all_data.iloc[train_size+dev_size:,:]


train_data.to_json('data/origin/intercontest/union_culture_kiwi_qa_error_postivate_train.json', force_ascii=False,orient='records', lines=True)


dev_data.to_json('data/origin/intercontest/union_culture_kiwi_qa_error_postivate_dev.json', force_ascii=False,orient='records', lines=True)


test_data.to_json('data/origin/intercontest/union_culture_kiwi_qa_error_postivate_test.json', force_ascii=False,orient='records', lines=True)