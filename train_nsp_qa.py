# -*- coding: utf-8 -*-

from functools import partial

import pandas as pd
import torch
import torch.utils.data as Data

# DataCollatorForLanguageModelingSpecial
from DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial
from PraticeOfTransformers.CustomModel import BertForUnionNspAndQA
from transformers import AutoTokenizer, BertForQuestionAnswering

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForUnionNspAndQA.from_pretrained(model_name, num_labels=2)

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
sent = ['我爱北京天安门，天安门上太阳升', '我爱北京中南海，毛主席在中南还','改革开放好，我哎深圳，深圳是改革开放先驱']
question = ['我爱什么?', '毛主席在哪?','谁是改革开放先驱']

# 创建一个实例，参数是tokenizer，如果不是batch的化，就采用tokenizer.encode_plus
encoded_dict = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=list(zip(sent, question)),  # 输入文本,采用list[tuple(question,text)]的方式进行输入 zip 把两个list压成tuple的list对
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
    text=list(text) #tuple 转为 list
    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]

    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''
    encoded_dict = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(questions,text)),  # 输入文本对 # 输入文本,采用list[tuple(question,text)]的方式进行输入
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=128,  # 填充 & 截断长度
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
    )
    input_ids = [torch.tensor(input_id ) for input_id in encoded_dict['input_ids']]
    attention_masks = [torch.tensor(attention ) for attention in encoded_dict['attention_mask']]
    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
    # output = data_collator(input_ids)
    output = data_collator(input_ids)
    return input_ids,attention_masks


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator)

train_dataloader = Data.DataLoader(
    passage_keyword_json, shuffle=True, collate_fn=create_batch_partial, batch_size=2
)

for returndata in train_dataloader:
    print(returndata)
