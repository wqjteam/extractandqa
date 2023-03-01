# -*- coding: utf-8 -*-
import os
import sys
from functools import partial

import numpy as np
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
from PraticeOfTransformers.CustomModelForBertWithoutNsp import CustomModelForBaseBertWithoutNsp
from PraticeOfTransformers.DataCollatorForWholeWordMaskOriginal import DataCollatorForWholeWordMaskOriginal
from PraticeOfTransformers.DataCollatorForWholeWordMaskSpecial import DataCollatorForWholeWordMaskSpecial

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定GPU编号 多gpu训练
model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomModelForBaseBertWithoutNsp.from_pretrained(model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递

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
# 获取遗址相关
passage_keyword_json_realrelic = pd.read_json("./data/origin/intercontest/passage_qa_keyword.json", orient='records',
                                              lines=True)

# 获取非遗址相关
passage_keyword_json_virtualrelic_file = open("./data/origin/intercontest/feiyi_passage_ner_withkeyword.txt",
                                              encoding="utf-8")
virtualrelic_setence = []
virtualrelic_keyword = []
for row in passage_keyword_json_virtualrelic_file.readlines():
    tmp_list = row.split(':')  # 按‘:'切分每行的数据
    virtualrelic_setence.append(row)
    virtualrelic_keyword.append([tmp_list[0]])
virtualrelic_setence_keyword = np.concatenate(
    (np.array(virtualrelic_setence).reshape(-1, 1), np.array(virtualrelic_keyword, dtype=list).reshape(-1, 1)), axis=1)
passage_keyword_json_virtualrelic = pd.DataFrame(data=virtualrelic_setence_keyword, columns=['sentence', 'keyword'])

union_pd = pd.concat(
    [passage_keyword_json_realrelic.loc[:, ['sentence', 'keyword']], passage_keyword_json_virtualrelic], axis=0)
train_data = union_pd.values

# train_data = train_data[:20]

_, dev_data = Data.random_split(union_pd.values, [int(len(union_pd.values) * 0.9),
                                                  len(union_pd.values) - int(
                                                      len(union_pd.values) * 0.9)])
# dev_data=dev_data[:2]

def create_batch(data, tokenizer, data_collator, keyword_flag=False):
    text, keyword = zip(*data)  # arrat的四列 转为tuple
    text = list(text)  # tuple 转为 list0
    keyword = [kw if isinstance(kw, list) else [kw] for kw in keyword]
    # questions = [q_a.get('question') for q_a in question_answer]

    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''

    encoded_dict_textandquestion = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=text,  # 不采用文本对，直接使用句子去训练
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=512,  # 填充 & 截断长度
        truncation=True,
        padding='longest',
        return_attention_mask=True,  # 返回 attn. masks.
    )

    encoded_dict_keywords = []

    for index, ks in enumerate(keyword):
        if len(ks) == 0:
            # print('没有keyword的句子:%s' % text[index])
            encoded_dict_keywords.append([])
        else:
            encoded_dict_keyword = tokenizer.batch_encode_plus(batch_text_or_text_pairs=ks,
                                                               add_special_tokens=True,
                                                               # 添加 '[CLS]' 和 '[SEP]' 需要添加，即使是不需要用到nsp的内容
                                                               pad_to_max_length=False,
                                                               return_attention_mask=False
                                                               )
            encoded_dict_keywords.append(encoded_dict_keyword['input_ids'])

    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
    base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict_textandquestion['input_ids']]
    attention_masks = [torch.tensor(attention) for attention in encoded_dict_textandquestion['attention_mask']]
    data_collator_output = data_collator(zip(base_input_ids, encoded_dict_keywords))
    # 由于不训练nsp了所以这里token_type_ids全部给0
    zero_token_type_ids = torch.zeros_like(torch.tensor(encoded_dict_textandquestion['token_type_ids']))
    mask_input_ids = data_collator_output["input_ids"]

    mask_input_labels = data_collator_output["labels"]  # 需要获取不是-100的位置，证明其未被替换，这也是target -100的位置在计算crossentropyloss 会丢弃

    # 对于model只接受tensor[list] 必须把 list[tensor] 转为tensor[list]
    return mask_input_ids, torch.stack(
        attention_masks), zero_token_type_ids, mask_input_labels, torch.tensor([]),


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, data_collator=data_collator)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

dev_dataloader = Data.DataLoader(
    dev_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

# 实例化相关metrics的计算对象   len(ner_id_label)+1是为了ignore_index用 他不允许为负数值 这里忽略了，所以不影响结果
model_recall = torchmetrics.Recall(num_classes=model.config.vocab_size + 1, mdmc_average='global',
                                   ignore_index=model.config.vocab_size)
model_precision = torchmetrics.Precision(num_classes=model.config.vocab_size + 1, mdmc_average='global',
                                         ignore_index=model.config.vocab_size)
model_f1 = torchmetrics.F1Score(num_classes=model.config.vocab_size + 1, mdmc_average='global',
                                ignore_index=model.config.vocab_size)

# 看是否用cpu或者gpu训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))

    model = nn.DataParallel(model, device_ids=device_ids)

model.to(device)

viz = Visdom(env=u'bert_base_special_train_without_nsp')  # 可视化
name = ['mlm_loss', 'total_loss']
name_precision_recall_f1 = ['precision_score', 'recall_score', 'f1_score']
viz.line(Y=[(0., 0.)], X=[(0., 0.)], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[(0., 0.)], X=[(0., 0.)], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置
viz.line(Y=[(0., 0., 0.)], X=[(0., 0., 0.)], win="pitcure_3",
         opts=dict(title='eval_precision_recall_f1', legend=name_precision_recall_f1, xlabel='epoch', ylabel='score',
                   markers=False))  # 绘制起始位置


#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch):
    eval_mlm_loss = 0
    # eval_nsp_loss = 0
    eval_total_loss = 0
    # eval_em_score = 0
    # eval_f1_score = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据
        mask_input_ids, attention_masks, token_type_ids, mask_input_labels, nsp_labels = return_batch_data

        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                             token_type_ids=token_type_ids.to(device))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        prediction_logits = model_output['prediction_logits'].to("cpu")
        # seq_relationship_logits = model_output['seq_relationship_logits'].to("cpu")

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100
        '''
        mlm loss 计算
        '''
        masked_lm_loss = loss_fct(prediction_logits.view(-1, model_config.vocab_size),
                                  mask_input_labels.view(-1))

        total_loss = masked_lm_loss

        predict = torch.argmax(prediction_logits, dim=2)
        mask_input_labels[mask_input_labels == -100] = model_config.vocab_size

        '''
        update 是计算当个batch的值  compute计算所有累加的值
        '''
        precision_score = model_precision(predict, mask_input_labels)
        model_precision.update(predict, mask_input_labels)
        recall_score = model_recall(predict, mask_input_labels)
        model_recall.update(predict, mask_input_labels)
        f1_score = model_f1(predict, mask_input_labels)
        model_f1.update(predict, mask_input_labels)

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
        # eval_nsp_loss += next_sentence_loss.detach()
        eval_total_loss += total_loss.detach()
        print('--eval---eopch: %d----precision_score: %f----recall_score: %f----f1_score: %f----mlm_loss: %f------ 损失函数: %.6f-' % (
            epoch, precision_score, recall_score, f1_score, masked_lm_loss, total_loss))

    viz.line(Y=[
        (eval_mlm_loss / eval_step, eval_total_loss / eval_step)], X=[(epoch + 1, epoch + 1)], win="pitcure_2",
        update='append')
    viz.line(Y=[(model_precision.compute().to('cpu'), model_recall.compute().to('cpu'), model_f1.compute().to('cpu'))],
             X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_3", update='append')
    model_precision.reset()
    model_recall.reset()
    model_f1.reset()

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
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        prediction_logits = model_output['prediction_logits'].to("cpu")
        # seq_relationship_logits = model_output['seq_relationship_logits'].to("cpu")

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
        # next_sentence_loss = loss_fct(seq_relationship_logits.view(-1, 2), nsp_labels.view(-1))

        total_loss = masked_lm_loss

        # 进行统计展示
        epoch_step += 1
        epoch_mlm_loss += masked_lm_loss.detach()
        # epoch_nsp_loss += next_sentence_loss.detach()

        epoch_total_loss += total_loss.detach()

        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        total_loss.backward()  # 反向传播
        print('第%d个epoch的%d批数据的mlm_loss: %f-----------的total_loss：%f' % (
            epoch + 1, step + 1, masked_lm_loss, total_loss))

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
    # numpy不可以直接在有梯度的数据上获取，需要先去除梯度
    # 绘制epoch以及对应的测试集损失loss 第一个参数是y  第二个是x
    viz.line(Y=[(epoch_mlm_loss / epoch_step, epoch_total_loss / epoch_step)], X=[(epoch + 1, epoch + 1)],
             win="pitcure_1", update='append')

    # 绘制评估函数相关数据
    evaluate(model=model, eval_data_loader=dev_dataloader, epoch=epoch)

    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(),
                   'save_model/bert_base_unionvitrual/bert_base_unionvitrual_epoch_%d' % (epoch + 1))

# 最后保存一下
torch.save(model.state_dict(), 'save_model/bert_base_unionvitrual/bert_base_unionvitrual_epoch_%d' % (epoch_size))
