# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Mapping, List, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import transformers
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel, BertForQuestionAnswering
from transformers.data.data_collator import _numpy_collate_batch, _torch_collate_batch, _tf_collate_batch, \
    DataCollatorMixin, DataCollatorForLanguageModeling
from functools import partial
# DataCollatorForLanguageModelingSpecial
import BertForUnionNspAndQA
from DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForUnionNspAndQA.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
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
sent = "我爱北京天安门，天安门上太阳升"
question = "我爱什么"
# 创建一个实例，参数是tokenizer
encoded_dict = tokenizer.encode_plus(
    sent,  # 输入文本
    question,  #
    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
    max_length=32,  # 填充 & 截断长度
    truncation=True,
    pad_to_max_length=True,
    return_attention_mask=True,  # 返回 attn. masks.
)

print(encoded_dict)
# input_ids = [torch.tensor(encoded_dict['input_ids'])]
# 传入的参数是tensor形式的input_ids，返回input_ids和label，label中
# -100的位置的词没有被mask，
# output = data_collator(input_ids)
# print(output)


def create_batch(data,tokenizer,data_collator):
    text, question_answer,keyword = zip(*data)
    questions=[q_a.get('question') for q_a in question_answer]
    answers=[q_a.get('answer') for q_a in question_answer]



    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''
    encoded_dict = tokenizer.encode_plus(
        questions,  # 输入文本
        answers,  #
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=32,  # 填充 & 截断长度
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
    )
    input_ids = [torch.tensor(encoded_dict['input_ids'])]
    output = data_collator(input_ids)
    return ('')

#把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator)

train_dataloader = Data.DataLoader(
    passage_keyword_json, shuffle=True, collate_fn=create_batch_partial, batch_size=10
)

for returndata in train_dataloader:
    print(returndata)
