﻿# -*- coding: utf-8 -*-
from functools import partial

import pandas as pd
import torch
import torch.utils.data as Data
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from transformers import AutoTokenizer

from PraticeOfTransformers.DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial
from PraticeOfTransformers.CustomModel import BertForUnionNspAndQA

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForUnionNspAndQA.from_pretrained(model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递
batch_size = 2
epoch_size = 10
# 用于梯度回归
optim = AdamW(model.parameters(), lr=5e-5)  # 需要填写模型的参数

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
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

passage_keyword_json = passage_keyword_json.explode("q_a").values

sent = ['我爱北京天安门，天安门上太阳升', '我爱北京中南海，毛主席在中南还', '改革开放好，我哎深圳，深圳是改革开放先驱']
question = ['我爱什么?', '毛主席在哪?', '谁是改革开放先驱']

train_data, dev_data = Data.random_split(passage_keyword_json, [int(len(passage_keyword_json) * 0.9),
                                                                len(passage_keyword_json) - int(
                                                                    len(passage_keyword_json) * 0.9)])

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

# print(encoded_dict)

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}


# print(output)


def create_batch(data, tokenizer, data_collator):
    text, question_answer, keyword, nsp = zip(*data)  # arrat的四列 转为tuple
    text = list(text)  # tuple 转为 list0
    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]
    nsps = list(nsp)  # tuple 转为list

    keywords = [kw[0] for kw in keyword]  # tuple 转为list 变成了双重的list 还是遍历转
    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置
    end_positions_labels = []  # 记录终止始位置
    for array_index, textstr in enumerate(text):
        start_in = textstr.find(answers[array_index])
        if start_in != -1 and nsps[array_index] == 1:  # 判断是否存在
            nsp_labels.append(nsp_label_id.get(True))
            start_positions_labels.append(start_in + 1)  # 因为在tokenizer.batch_encode_plus中转换的时候添加了cls
            end_positions_labels.append(start_in + 1 + len(answers[array_index]))
        else:
            nsp_labels.append(nsp_label_id.get(False))
            # 若找不到，则将start得位置 放在最末尾的位置 pad或者 [SEP]
            start_positions_labels.append(
                len(textstr))  # 应该是len(textstr) -1得 但是因为在tokenizer.batch_encode_plus中转换的时候添加了cls 所以就是len(textstr) -1 +1
            end_positions_labels.append(len(textstr))

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
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
    )
    encoded_dict_keywords = tokenizer.batch_encode_plus(batch_text_or_text_pairs=keywords, add_special_tokens=False,
                                                        # 添加 '[CLS]' 和 '[SEP]'
                                                        pad_to_max_length=False,
                                                        return_attention_mask=False
                                                        )
    base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict_textandquestion['input_ids']]
    attention_masks = [torch.tensor(attention) for attention in encoded_dict_textandquestion['attention_mask']]
    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
    data_collator_output = data_collator(zip(base_input_ids, encoded_dict_keywords['input_ids']))
    mask_input_ids = data_collator_output["input_ids"]

    mask_input_labels = data_collator_output["labels"]  # 需要获取不是-100的位置，证明其未被替换，这也是target -100的位置在计算crossentropyloss 会丢弃
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

    # 对于model只接受tensor[list] 必须为 list[tensor] 转为tensor[list]
    return mask_input_ids, torch.stack(
        attention_masks), mask_input_labels, torch.tensor(nsp_labels), torch.tensor(
        start_positions_labels), torch.tensor(end_positions_labels)


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

dev_dataloader = Data.DataLoader(
    dev_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

# 实例化相关metrics的计算对象
model_recall = torchmetrics.Recall(task='binary', average='macro', num_classes=2)
model_precision = torchmetrics.Precision(task='binary', average='macro', num_classes=2)
model_f1 = torchmetrics.F1Score(task='binary', average="macro", num_classes=2)

# 看是否用cpu或者gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)
model.to(device)


#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, data_loader):
    # 依次处理每批数据
    for return_batch_data in train_dataloader:  # 一个batch一个bach的训练完所有数据
        mask_input_ids, mask_input_labels, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data
        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device))
        # 进行转换

        config = model.config
        prediction_scores = model_output.mlm_prediction_scores
        nsp_relationship_scores = model_output.nsp_relationship_scores
        qa_start_logits = model_output.qa_start_logits
        qa_end_logits = model_output.qa_end_logits

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100
        '''
        mlm loss 计算
        '''
        mlm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), mask_input_labels.view(-1))

        '''
        nsp loss 计算
        '''
        nsp_loss = loss_fct(nsp_relationship_scores.view(-1, 2), nsp_labels.view(-1))

        '''
        qa loss 计算
        '''
        start_loss = loss_fct(qa_start_logits, start_positions_labels)
        end_loss = loss_fct(qa_end_logits, end_positions_labels)
        qa_loss = (start_loss + end_loss) / 2

        total_loss = mlm_loss + torch.exp(nsp_loss) * qa_loss

        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        total_loss.backward()  # 反向传播
        print(total_loss)

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
        # 损失函数的平均值

        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始

    f1_score = model_f1.compute()
    recall = model_recall.compute()
    precision = model_precision.compute()

    # 清空计算对象
    model_precision.reset()
    model_f1.reset()
    model_recall.reset()
    print("评估准确度: %.6f - 召回率: %.6f - f1得分: %.6f- 损失函数: %.6f" % (precision, recall, f1_score, total_loss))


# 进行训练
for epoch in range(epoch_size):  # 所有数据迭代总的次数

    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据
        mask_input_ids, attention_masks, mask_input_labels, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data
        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device))
        config = model.config
        prediction_scores = model_output.mlm_prediction_scores
        nsp_relationship_scores = model_output.nsp_relationship_scores
        qa_start_logits = model_output.qa_start_logits
        qa_end_logits = model_output.qa_end_logits

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100
        '''
        mlm loss 计算
        '''
        mlm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), mask_input_labels.view(-1))

        '''
        nsp loss 计算
        '''
        nsp_loss = loss_fct(nsp_relationship_scores.view(-1, 2), nsp_labels.view(-1))

        '''
        qa loss 计算
        '''
        start_loss = loss_fct(qa_start_logits, start_positions_labels)
        end_loss = loss_fct(qa_end_logits, end_positions_labels)
        qa_loss = (start_loss + end_loss) / 2

        total_loss = mlm_loss + torch.exp(nsp_loss) * qa_loss  # 目的是为了当nsp预测错了的时候 加大惩罚程度

        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        total_loss.backward()  # 反向传播
        print("第%d个epoch的%d批数据的loss：%f"(epoch, step, total_loss))

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作

torch.save(model.state_dict(), "save_model/path1")
