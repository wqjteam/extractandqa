# -*- coding: utf-8 -*-
import os
import sys
from functools import partial

import pandas as pd
import torch
import torch.utils.data as Data
import torchmetrics
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForPreTraining
from visdom import Visdom

from PraticeOfTransformers import Utils
from PraticeOfTransformers.DataCollatorForWholeWordMaskOriginal import DataCollatorForWholeWordMaskOriginal
from PraticeOfTransformers.DataCollatorForWholeWordMaskSpecial import DataCollatorForWholeWordMaskSpecial

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForPreTraining.from_pretrained(model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递

batch_size = 2
epoch_size = 1000
# 用于梯度回归
optim = Adam(model.parameters(), lr=5e-5)  # 需要填写模型的参数
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

'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

passage_keyword_json = passage_keyword_json.explode("q_a").values



train_data, dev_data = Data.random_split(passage_keyword_json, [int(len(passage_keyword_json) * 0.9),
                                                                len(passage_keyword_json) - int(
                                                                    len(passage_keyword_json) * 0.9)])

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}


def create_batch(data, tokenizer, data_collator, keyword_flag=False):
    text, question_answer, keyword, nsp = zip(*data)  # arrat的四列 转为tuple
    text = list(text)  # tuple 转为 list0
    questions = [q_a.get('question') for q_a in question_answer]

    nsps = list(nsp)  # tuple 转为list

    keywords = [kw[0] for kw in keyword]  # tuple 转为list 变成了双重的list 还是遍历转
    nsp_labels = nsps  # 用作判断两句是否相关

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

    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
    base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict_textandquestion['input_ids']]
    attention_masks = [torch.tensor(attention) for attention in encoded_dict_textandquestion['attention_mask']]
    data_collator_output = data_collator(zip(base_input_ids, encoded_dict_keywords['input_ids']))
    mask_input_ids = data_collator_output["input_ids"]

    mask_input_labels = data_collator_output["labels"]  # 需要获取不是-100的位置，证明其未被替换，这也是target -100的位置在计算crossentropyloss 会丢弃

    # 对于model只接受tensor[list] 必须把 list[tensor] 转为tensor[list]
    return mask_input_ids, torch.stack(
        attention_masks), torch.tensor(
        encoded_dict_textandquestion['token_type_ids']), mask_input_labels, torch.tensor(nsp_labels),


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

dev_dataloader = Data.DataLoader(
    dev_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)



# 看是否用cpu或者gpu训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)

if torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))
    print('111111111111111111111111111111111')
    model = nn.DataParallel(model,device_ids=device_ids)
print('22222222222222222222222222222222222')
model.to(device)

viz = Visdom(env=u'bert_base_special_train')  # 可视化
name = ['mlm_loss', 'nsp_loss', 'total_loss']
viz.line(Y=[(0., 0., 0.)], X=[(0., 0., 0.)], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[(0., 0., 0.)], X=[(0., 0., 0.)], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置


#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch):
    eval_mlm_loss = 0
    eval_nsp_loss = 0
    eval_total_loss = 0
    eval_em_score = 0
    eval_f1_score = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据
        mask_input_ids, attention_masks, token_type_ids, mask_input_labels, nsp_labels = return_batch_data

        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                             token_type_ids=token_type_ids.to(device))
        if torch.cuda.device_count() > 1:
            model_config=model.module.config
        else:
            model_config = model.config

        prediction_logits = model_output['prediction_logits'].to("cpu")
        seq_relationship_logits = model_output['seq_relationship_logits'].to("cpu")

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100
        '''
        mlm loss 计算
        '''
        masked_lm_loss = loss_fct(prediction_logits.view(-1, model_config.vocab_size),
                                  mask_input_labels.view(-1))

        '''
        nsp loss 计算
        '''
        next_sentence_loss = loss_fct(seq_relationship_logits.view(-1, 2), nsp_labels.view(-1))

        total_loss = masked_lm_loss + next_sentence_loss

        '''
        实际预测值与目标值的
        '''
        # qa_start_logits_argmax = torch.argmax(seq_relationship_score, dim=1)
        # qa_end_logits_argmax = torch.argmax(prediction_logits, dim=1)

        # 损失函数的平均值
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        # 进行统计展示
        eval_step += 1
        eval_mlm_loss += masked_lm_loss.detach()
        eval_nsp_loss += next_sentence_loss.detach()
        eval_total_loss += total_loss.detach()
        print('--eval---eopch: %d----mlm_loss: %f----的nsp_loss: %f------ 损失函数: %.6f' % (epoch, masked_lm_loss, next_sentence_loss , total_loss ))

    viz.line(Y=[
        (eval_mlm_loss / eval_step, eval_nsp_loss / eval_step, eval_total_loss / eval_step)],
        X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_2", update='append')






# 进行训练
for epoch in range(epoch_size):  # 所有数据迭代总的次数
    epoch_mlm_loss = 0
    epoch_nsp_loss = 0
    epoch_total_loss = 0
    epoch_step = 0
    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据

        mask_input_ids, attention_masks, token_type_ids, mask_input_labels, nsp_labels = return_batch_data

        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                             token_type_ids=token_type_ids.to(device))
        if torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        prediction_logits = model_output['prediction_logits'].to("cpu")
        seq_relationship_logits = model_output['seq_relationship_logits'].to("cpu")

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100
        '''
        mlm loss 计算
        '''
        masked_lm_loss = loss_fct(prediction_logits.view(-1, model_config.vocab_size),
                                  mask_input_labels.view(-1))

        '''
        nsp loss 计算
        '''
        next_sentence_loss = loss_fct(seq_relationship_logits.view(-1, 2), nsp_labels.view(-1))

        total_loss = masked_lm_loss + next_sentence_loss

        # 进行统计展示
        epoch_step += 1
        epoch_mlm_loss += masked_lm_loss.detach()
        epoch_nsp_loss += next_sentence_loss.detach()

        epoch_total_loss += total_loss.detach()

        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        total_loss.backward()  # 反向传播
        print('第%d个epoch的%d批数据的mlm_loss: %f------的nsp_loss: %f-----的total_loss：%f' % (
        epoch + 1, step + 1, masked_lm_loss, next_sentence_loss, total_loss))

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
    # numpy不可以直接在有梯度的数据上获取，需要先去除梯度
    # 绘制epoch以及对应的测试集损失loss 第一个参数是y  第二个是x
    viz.line(Y=[(epoch_mlm_loss / epoch_step, epoch_nsp_loss / epoch_step,
                 epoch_total_loss / epoch_step)],
             X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_1", update='append')

    # 绘制评估函数相关数据
    evaluate(model=model, eval_data_loader=dev_dataloader, epoch=epoch)

    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), 'save_model/bert_base/bert_base_epoch_%d' % (epoch + 1))

# 最后保存一下
torch.save(model.state_dict(), 'save_model/bert_base/bert_base_epoch_%d' % (epoch_size))
