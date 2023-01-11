# -*- coding: utf-8 -*-
import copy
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

# DataCollatorForLanguageModelingSpecial
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from PraticeOfTransformers.Utils import pad_sequense_python
from DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial
from PraticeOfTransformers.CustomModel import BertForUnionNspAndQA
from transformers import AutoTokenizer, BertForQuestionAnswering

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForUnionNspAndQA.from_pretrained(model_name, num_labels=2)  #num_labels 测试用一下

# model = BertForUnionNspAndQA.from_pretrained(model_name)
print(model)
# data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
#                                                              mlm=True,
#                                                              mlm_probability=0.15,
#                                                              return_tensors="pt")

data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
                                                       mlm=True,
                                                       mlm_probability=0.15,
                                                       return_tensors="pt")

'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword.json", orient='records',
                                    lines=True).head(100).drop("spos", axis=1)
passage_keyword_json = passage_keyword_json.explode("q_a").values
sent = ['我爱北京天安门，天安门上太阳升', '我爱北京中南海，毛主席在中南还', '改革开放好，我哎深圳，深圳是改革开放先驱']
question = ['我爱什么?', '毛主席在哪?', '谁是改革开放先驱']

# 创建一个实例，参数是tokenizer，如果不是batch的化，就采用tokenizer.encode_plus
encoded_dict = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=list(zip(sent, question)),
    # 输入文本,采用list[tuple(question,text)]的方式进行输入 zip 把两个list压成tuple的list对
    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
    max_length=128,  # 填充 & 截断长度
    truncation=True,
    pad_to_max_length=True,
    return_attention_mask=True,  # 返回 attn. masks.
)

print(encoded_dict)


# print(output)


def create_batch(data, tokenizer, data_collator):
    text, question_answer, keyword = zip(*data)
    text = list(text)  # tuple 转为 list
    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]
    nsp = []  # 用作判断两句是否相关
    start_positions = []  # 记录起始位置
    end_positions = []  # 记录终止始位置
    for array_index, textstr in enumerate(text):
        start_in = textstr.find(answers[array_index])
        if start_in != -1:  # 判断是否存在
            nsp.append(True)
            start_positions.append(start_in + 1)  # 因为在tokenizer.batch_encode_plus中转换的时候添加了cls
            end_positions.append(start_in + 1 + len(answers[array_index]))
        else:
            nsp.append(False)
            start_positions.append(-1)
            end_positions.append(-1)

    # start_positions = [q_a.get('start_position') for q_a in question_answer]
    # end_positions = [q_a.get('end_position') for q_a in question_answer]
    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''
    encoded_dict = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(text, questions)),  # 输入文本对 # 输入文本,采用list[tuple(text,question)]的方式进行输入
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=512,  # 填充 & 截断长度
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
    )
    base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict['input_ids']]
    attention_masks = [torch.tensor(attention) for attention in encoded_dict['attention_mask']]
    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
    data_collator_output = data_collator(base_input_ids)
    mask_input_ids = data_collator_output["input_ids"]



    '''
     进行遍历,获取原来的未被mask的数据,再对此进行对齐,方便后续会对此进行计算loss
    '''
    mask_input_labels = data_collator_output["labels"]  # 需要获取不是-100的位置，证明其未被替换
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


    # 对于model只接受tensor[list] 必须为 list[tensor] 转为tensor[list]
    return mask_input_ids , torch.stack(attention_masks), mask_input_postion,mask_input_value, nsp, start_positions, end_positions


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator)

train_dataloader = Data.DataLoader(
    passage_keyword_json, shuffle=True, collate_fn=create_batch_partial, batch_size=2
)

for returndata in train_dataloader:
    mask_input_ids, attention_masks, mask_input_postion,mask_input_value, nsp, start_positions, end_positions = returndata
    model_output=model(input_ids=mask_input_ids, attention_mask=attention_masks)
    print(model_output)
