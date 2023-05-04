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

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
data_collator = DataCollatorForWholeWordMaskSpecial(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.15,
                                                    return_tensors="pt")
if len(sys.argv) >= 4 and sys.argv[3] == 'origin':
    data_collator = DataCollatorForWholeWordMaskOriginal(tokenizer=tokenizer,
                                                         mlm=True,
                                                         mlm_probability=0.15,
                                                         return_tensors="pt")

keyword_flag = False
if len(sys.argv) >= 5 and sys.argv[4] == 'True':
    keyword_flag = True

# 获取模型路径
if len(sys.argv) >= 6:
    print('-------------load para------------------%s--------------' % sys.argv[5])
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = BertForUnionNspAndQA(config)

    state_dict = torch.load(sys.argv[5])
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():  # k为module.xxx.weight, v为权重
        if k.startswith('module.'):
            name = k[7:]  # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # 因为后面的参数没有初始化，所以采用非强制性约束
    model.load_state_dict(new_state_dict, strict=False)
else:
    ##完全继承
    model = BertForUnionNspAndQA.from_pretrained(model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递


'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)
# passage_keyword_json['q_a'] 和 passage_keyword_json['q_a'].q_a 一样
passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

# passage_keyword_json = passage_keyword_json[passage_keyword_json.nsp == 1]
# passage_keyword_json = passage_keyword_json[passage_keyword_json['sentence'].apply(lambda x: '长治市博物馆，' in x)]

passage_keyword_json = passage_keyword_json.explode("q_a").values

sent = ['我爱北京天安门，天安门上太阳升', '我爱北京中南海，毛主席在中南还', '改革开放好，我哎深圳，深圳是改革开放先驱']
question = ['我爱什么?', '毛主席在哪?', '谁是改革开放先驱']

train_data, dev_data = Data.random_split(passage_keyword_json, [int(len(passage_keyword_json) * 0.9),
                                                                len(passage_keyword_json) - int(
                                                                    len(passage_keyword_json) * 0.9)],
                                         generator=torch.Generator().manual_seed(0))

# 创建一个实例，参数是tokenizer，如果不是batch的化，就采用tokenizer.encode_plus
encoded_dict = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=list(zip(sent, question)),
    # 输入文本,采用list[tuple(question,text)]的方式进行输入 zip 把两个list压成tuple的list对
    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
    max_length=128,  # 填充 & 截断长度
    truncation=True,
    padding='longest',
    return_attention_mask=True,  # 返回 attn. masks.
)

# print(encoded_dict)

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}


# print(output)


def create_batch(data, tokenizer, data_collator, keyword_flag=False):
    text, question_answer, keyword, nsp = zip(*data)  # arrat的四列 转为tuple
    text = list(text)  # tuple 转为 list0
    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]
    nsps = list(nsp)  # tuple 转为list

    keywords = [kw[0] for kw in keyword]  # tuple 转为list 变成了双重的list 还是遍历转
    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置 需要在进行encode之后再进行添加
    end_positions_labels = []  # 记录终止始位置 需要在进行encode之后再进行添加

    # start_positions = [q_a.get('start_position') for q_a in question_answer]
    # end_positions = [q_a.get('end_position') for q_a in question_answer]
    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''

    encoded_dict_textandquestion = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(text, questions)),  # 输入文本对 # 输入文本,采用list[tuple(text,question)]的方式进行输入
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=512,  # 填充 & 截断长度
        truncation=True,
        padding='longest',
        return_attention_mask=True,  # 返回 attn. masks.
    )

    encoded_dict_keywords = tokenizer.batch_encode_plus(batch_text_or_text_pairs=keywords,
                                                        add_special_tokens=False,  # 添加 '[CLS]' 和 '[SEP]'
                                                        pad_to_max_length=False,
                                                        return_attention_mask=False
                                                        )
    '''
    处理qa的问题
    '''
    encoded_dict_answers = tokenizer.batch_encode_plus(batch_text_or_text_pairs=answers,
                                                       add_special_tokens=False,  # 添加 '[CLS]' 和 '[SEP]'
                                                       pad_to_max_length=False,
                                                       return_attention_mask=False
                                                       )

    for array_index, textstr in enumerate(encoded_dict_textandquestion['input_ids']):

        start_in = CommonUtil.get_first_index_in_array(textstr, encoded_dict_answers['input_ids'][
            array_index])  # 这方法在data_collator存在，不再重复写了
        if start_in != -1 and nsps[array_index] == 1:  # 判断是否存在
            nsp_labels.append(nsp_label_id.get(True))
            start_positions_labels.append(start_in)  # 因为在tokenizer.batch_encode_plus中转换的时候添加了cls
            end_positions_labels.append(start_in + 1 + len(answers[array_index]))
        else:
            nsp_labels.append(nsp_label_id.get(False))
            # 若找不到，则将start得位置 放在最末尾的位置 padding 或者 [SEP]

            # 应该是len(textstr) -1得 但是因为在tokenizer.batch_encode_plus中转换的时候添加了cls 所以就是len(textstr) -1 +1
            sep_in = textstr.index(tokenizer.encode(text='[SEP]', add_special_tokens=False)[0])
            start_positions_labels.append(sep_in)
            end_positions_labels.append(sep_in)

    if keyword_flag:
        # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
        base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict_textandquestion['input_ids']]
        attention_masks = [torch.tensor(attention) for attention in encoded_dict_textandquestion['attention_mask']]
        data_collator_output = data_collator(zip(base_input_ids, encoded_dict_keywords['input_ids']))
        mask_input_ids = data_collator_output["input_ids"]

        mask_input_labels = data_collator_output[
            "labels"]  # 需要获取不是-100的位置，证明其未被替换，这也是target -100的位置在计算crossentropyloss 会丢弃
        '''
         进行遍历,获取原来的未被mask的数据,再对此进行对齐,方便后续会对此进行计算loss 在计算loss的时候 根据CrossEntropyLoss 的ingore_index 会根据target的属性，
         直接去出某些值.不进行计算，所以不需要再进行转换
        '''
        '''
        mask_input_postion_x, mask_input_postion_y = torch.where(
            mask_input_labels != -100)  # 二维数据，结果分为2个array,按照15的% mask 每行都会有此呗mask
        # mask_input_postion = torch.reshape(mask_input_postion_y, (mask_input_labels.shape[0], -1))  # -1表示不指定 自己去推测,所有的mask的数据必须等长，方便后续的loss使用矩阵计算
        mask_input_postion_y=mask_input_postion_y.numpy() #转为np好计算，不然tensor 中套tensor
        mask_input_postion=[]
        mask_input_value=[]
        forntindex=0
        rearfront=0
        rowindex=0
        while(forntindex<len(mask_input_postion_x)):
            while(forntindex<len(mask_input_postion_x) and mask_input_postion_x[forntindex] ==mask_input_postion_x[rearfront]):
                forntindex+=1
            #获取位置
            y_index=list(mask_input_postion_y[rearfront:forntindex])
            mask_input_postion.append(y_index)
            mask_input_value.append([base_input_ids[rowindex].numpy()[colindex] for colindex in y_index])
            rearfront=forntindex
            rowindex+=1
        mask_input_postion=pad_sequense_python(mask_input_postion,-1) #后续计算的时候对于-1不进行计算
        mask_input_value=pad_sequense_python(mask_input_value,-1)
         '''

        # 对于model只接受tensor[list] 必须把 list[tensor] 转为tensor[list]
        return mask_input_ids, torch.stack(
            attention_masks), torch.tensor(
            encoded_dict_textandquestion['token_type_ids']), mask_input_labels, torch.tensor(nsp_labels), torch.tensor(
            start_positions_labels), torch.tensor(end_positions_labels)
    else:
        # mask_input_labels 位置用torch.tensor(encoded_dict_textandquestion['input_ids'])代替了，这里不计算mlm的loss了
        return torch.tensor(encoded_dict_textandquestion['input_ids']), torch.tensor(
            encoded_dict_textandquestion['attention_mask']), \
               torch.tensor(encoded_dict_textandquestion['token_type_ids']), torch.tensor(
            encoded_dict_textandquestion['input_ids']), torch.tensor(nsp_labels), torch.tensor(
            start_positions_labels), torch.tensor(end_positions_labels)


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator,
                               keyword_flag=keyword_flag)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)
