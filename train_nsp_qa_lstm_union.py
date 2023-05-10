# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup
from visdom import Visdom

import CommonUtil
import data_get_qa_all_label
from PraticeOfTransformers import Utils
from PraticeOfTransformers.CustomModelForNSPQABILSTM import CustomModelForNSPQABILSTM

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

keyword_flag = False
if len(sys.argv) >= 4 and sys.argv[3] == 'True':
    keyword_flag = True

# 获取模型路径
if len(sys.argv) >= 5:

    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = CustomModelForNSPQABILSTM(config)
    # 因为后面的参数没有初始化，所以采用非强制性约束
    model.load_state_dict(torch.load(sys.argv[4]), strict=False)
else:
    ##完全继承
    model = CustomModelForNSPQABILSTM.from_pretrained(model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递

# 用于梯度回归


'''
获取数据
'''
train_data = pd.read_json("./data/origin/intercontest/union_culture_kiwi_qa_error_postivate_train.json",
                          orient='records',
                          lines=True)
dev_data = pd.read_json("./data/origin/intercontest/union_culture_kiwi_qa_error_postivate_dev.json", orient='records',
                        lines=True)
test_data = pd.read_json("./data/origin/intercontest/union_culture_kiwi_qa_error_postivate_test.json", orient='records',
                         lines=True)
train_data = train_data[train_data['nsp'].apply(lambda x: x==0)]



train_data = train_data[train_data['passage'].apply(lambda x: x.startswith('J Storm') )]
# passage_keyword_json = passage_keyword_json[:10]


def tokenize_and_align_labels(data, tokenizer):
    _, text, q_a, nsp = zip(*data)  # arrat的四列 转为tuple
    textarray=[]
    for data in text:
        if len(data) > 512 - 3 - 30:  # 给问题留30个字
            textarray.append(list(data[:512 - 3 - 30]))
        else:
            textarray.append(list(data))
    # text = [list(data) for data in text]  # tuple 转为 list0
    questions = [list(qa.get('question')) for qa in q_a]
    answers = [qa.get('answer').get('text') for qa in q_a]
    answers_index = [(qa.get('answer').get('start'), qa.get('answer').get('end')) for qa in q_a]
    nsps = list(nsp)  # tuple 转为list

    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置 需要在进行encode之后再进行添加
    end_positions_labels = []  # 记录终止始位置 需要在进行encode之后再进行添加

    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''

    encoded_dict_textandquestion = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(textarray, questions)),  # 输入文本对 # 输入文本,采用list[tuple(text,question)]的方式进行输入
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=512,  # 填充 & 截断长度
        truncation=True,
        padding='longest',
        return_attention_mask=True,  # 返回 attn. masks.
        is_split_into_words=True
    )

    for index in range(len(textarray)):
        # 获取subtokens位置
        passage_len = len(textarray[index])
        start_index, end_index = answers_index[index]
        true_start_index = -1
        true_end_index = -1
        word_ids = encoded_dict_textandquestion.word_ids(batch_index=index)
        previous_word_idx = -1
        label_ids = []
        true_index = -1
        '''
        如果nsp是0的话，代表代表没有答案
        '''
        if nsps[index] == 0 or start_index > passage_len or end_index > passage_len:
            position_index = passage_len + 1
            answers_index[index] = (position_index, position_index)
            continue
        # 遍历subtokens位置索引
        for word_index, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                pass
                # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                true_index += 1
                if true_index == start_index:

                    true_start_index = word_index
                elif true_index == end_index:

                    true_end_index = word_index
            if true_end_index != -1:
                answers_index[index] = (true_start_index, true_end_index)
                break
            previous_word_idx = word_idx

    return encoded_dict_textandquestion, answers_index


train_data = train_data.explode("q_a").values
dev_data = dev_data.explode("q_a").values
test_data = test_data.explode("q_a").values

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}

# print(output)
# 用于梯度回归
if full_fine_tuning:
    # model.named_parameters(): [bert, bilstm, classifier, crf]
    bert_optimizer = list(model.bert.named_parameters())
    lstm_optimizer = list(model.bilstm.named_parameters())
    nsp_cls_optimizer = list(model.nsp_cls.named_parameters())
    pooler_optimizer = list(model.pooler.named_parameters())
    qa_outputs_optimizer = list(model.qa_outputs.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 5, 'weight_decay': weight_decay},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],  # 对于在no_decay 不进行正则化
         'lr': learning_rate * 5, 'weight_decay': 0.0},
        {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 100, 'weight_decay': weight_decay},
        {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
         'lr': learning_rate * 100, 'weight_decay': 0.0},
        {'params': [p for n, p in nsp_cls_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 10, 'weight_decay': weight_decay},
        {'params': [p for n, p in nsp_cls_optimizer if any(nd in n for nd in no_decay)],
         'lr': learning_rate * 10, 'weight_decay': 0.0},
        {'params': [p for n, p in pooler_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 10, 'weight_decay': weight_decay},
        {'params': [p for n, p in pooler_optimizer if any(nd in n for nd in no_decay)],
         'lr': learning_rate * 10, 'weight_decay': 0.0},
        {'params': [p for n, p in qa_outputs_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 10, 'weight_decay': weight_decay},
        {'params': [p for n, p in qa_outputs_optimizer if any(nd in n for nd in no_decay)],
         'lr': learning_rate * 10, 'weight_decay': 0.0},

    ]
    # only fine-tune the head classifier
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
optim = AdamW(optimizer_grouped_parameters, lr=learning_rate)
train_steps_per_epoch = len(train_data) // batch_size
scheduler = get_cosine_schedule_with_warmup(optim,
                                            num_warmup_steps=(epoch_size // 10) * train_steps_per_epoch,
                                            num_training_steps=epoch_size * train_steps_per_epoch)


def create_batch(data, tokenizer):
    _, text, q_a, nsp = zip(*data)  # arrat的四列 转为tuple
    text = list(text)  # tuple 转为 list0
    questions = [qa.get('question') for qa in q_a]
    answers = [qa.get('answer').get('text') for qa in q_a]
    source_answers_index = [(qa.get('answer').get('start'), qa.get('answer').get('end')) for qa in q_a]
    text_qa_tokens, answers_index = tokenize_and_align_labels(data, tokenizer)
    nsps = list(nsp)  # tuple 转为list

    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置 需要在进行encode之后再进行添加
    end_positions_labels = []  # 记录终止始位置 需要在进行encode之后再进行添加

    aa = list(zip(*answers_index))
    # mask_input_labels 位置用torch.tensor(encoded_dict_textandquestion['input_ids'])代替了，这里不计算mlm的loss了
    return torch.tensor(text_qa_tokens['input_ids']), torch.tensor(
        text_qa_tokens['attention_mask']), \
           torch.tensor(text_qa_tokens['token_type_ids']), torch.tensor(nsps), torch.tensor(
        list(zip(*answers_index))[0]), torch.tensor(list(zip(*answers_index))[1])


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

dev_dataloader = Data.DataLoader(
    dev_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

test_dataloader = Data.DataLoader(
    test_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

# 看是否用cpu或者gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)

if torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))

    model = nn.DataParallel(model, device_ids=device_ids)

model.to(device)
'''
可视化
'''
viz = Visdom(env=u'qa_%s_lstm_train' % model_name)
name = ['nsp_loss', 'qa_loss', 'total_loss']
name_em_f1 = ['em_score', 'f1_score']
viz.line(Y=[(0., 0., 0.)], X=[(0., 0., 0.)], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[(0., 0., 0.)], X=[(0., 0., 0.)], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置
viz.line(Y=[(0., 0.)], X=[(0., 0.)], win="pitcure_3",
         opts=dict(title='eval_em_f1', legend=name_em_f1, xlabel='epoch', ylabel='score(100分制)',
                   markers=False))  # 绘制起始位置


#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch, tokenizer):
    eval_nsp_loss = 0
    eval_qa_loss = 0
    eval_total_loss = 0
    eval_em_score = 0
    eval_f1_score = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据
        mask_input_ids, attention_masks, token_type_ids, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data
        with torch.no_grad():
            model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                                 token_type_ids=token_type_ids.to(device))

        # 进行转换
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        nsp_relationship_scores = model_output.nsp_relationship_scores.to("cpu")
        qa_start_logits = model_output.qa_start_logits.to("cpu")
        qa_end_logits = model_output.qa_end_logits.to("cpu")

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100

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

        total_loss = torch.sqrt(torch.exp(nsp_loss)) * qa_loss  # 目的是为了当nsp预测错了的时候 加大惩罚程度
        # total_loss = torch.sqrt(torch.exp(nsp_loss)) * qa_loss  # 目的是为了当nsp预测错了的时候 加大惩罚程度

        '''
        实际预测值与目标值的
        '''
        qa_start_logits_argmax = torch.argmax(qa_start_logits, dim=1)
        qa_end_logits_argmax = torch.argmax(qa_end_logits, dim=1)
        qa_predict = [Utils.get_all_word(tokenizer, mask_input_ids[index, start:end].numpy().tolist()) for
                      index, (start, end) in enumerate(zip(qa_start_logits_argmax, qa_end_logits_argmax))]
        qa_target = [Utils.get_all_word(tokenizer, mask_input_ids[index, start:end].numpy().tolist()) for
                     index, (start, end) in enumerate(zip(start_positions_labels, end_positions_labels))]
        qa_metric = Utils.get_eval(pred_arr=qa_predict, target_arr=qa_target)
        em_score = qa_metric['EM']
        f1_score = qa_metric['F1']

        print('--eval---epoch次数:%d---em得分: %.6f - f1得分: %.6f--nsp损失函数: %.6f--qa损失函数: %.6f- total损失函数: %.6f'
              % (epoch, em_score, f1_score, nsp_loss, qa_loss, total_loss.detach()))
        # 损失函数的平均值
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        # 进行统计展示
        eval_step += 1

        eval_nsp_loss += nsp_loss.detach()
        eval_qa_loss += qa_loss.detach()
        eval_total_loss += total_loss.detach()
        eval_em_score += em_score
        eval_f1_score += f1_score

    viz.line(Y=[
        (eval_nsp_loss / eval_step, eval_qa_loss / eval_step, eval_total_loss / eval_step)],
        X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_2", update='append')
    viz.line(Y=[(eval_em_score / eval_step, eval_f1_score / eval_step)],
             X=[(epoch + 1, epoch + 1)], win="pitcure_3", update='append')
    if eval_em_score / eval_step >= 74.3:
        # if eval_em_score / eval_step >= 0.2:
        encoded_dict = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=list(
                zip(["春秋版画博物馆是坐落于北京的一所主要藏品为版画的博物馆。"], ["春秋版画博物馆在哪里？"])),
            # 输入文本对 # 输入文本,采用list[tuple(text,question)]的方式进行输入
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=512,  # 填充 & 截断长度
            truncation=True,
            padding='longest',
            return_attention_mask=True,  # 返回 attn. masks.
        )
        model_output = model(input_ids=torch.tensor(encoded_dict['input_ids']).to(device),
                             attention_mask=torch.tensor(encoded_dict['attention_mask']).to(device),
                             token_type_ids=torch.tensor(encoded_dict['token_type_ids']).to(device))
        qa_start_logits = model_output.qa_start_logits.to("cpu")
        qa_end_logits = model_output.qa_end_logits.to("cpu")
        inputlen = len(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))
        rownames = [str(index) + '-' + data for index, data in
                    enumerate(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))]

        viz.bar(X=qa_start_logits.view(-1)[:inputlen].tolist(), win="pitcure_4",
                opts=dict(title='start_word_sorce',
                          stacked=False,
                          rownames=rownames,
                          xlabel='word',
                          ylabel='score',
                          markers=False))
        viz.bar(X=qa_end_logits[:inputlen].tolist(), win="pitcure_5",
                opts=dict(title='end_word_sorce',
                          stacked=False,
                          rownames=rownames,
                          xlabel='word',
                          ylabel='score',
                          markers=False))

        qa_start_logits_argmax = torch.argmax(qa_start_logits, dim=1)
        qa_end_logits_argmax = torch.argmax(qa_end_logits, dim=1)
        qa_predict = [
            Utils.get_all_word(tokenizer, torch.tensor(encoded_dict['input_ids'])[index, start:end].numpy().tolist())
            for
            index, (start, end) in enumerate(zip(qa_start_logits_argmax, qa_end_logits_argmax))]
        torch.save(model.state_dict(), 'save_model/nsp_qa_lstm/ultimate_nsp_qa_lstm_epoch_%d' % (epoch_size))
        print(''.join(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0])))
        print(qa_predict)
        print("结束！！！！！！！！！！！！！！！！")
        sys.exit(0)


# 进行训练
model.train()
for epoch in range(epoch_size):  # 所有数据迭代总的次数

    epoch_nsp_loss = 0
    epoch_qa_loss = 0
    epoch_total_loss = 0
    epoch_step = 0
    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据

        mask_input_ids, attention_masks, token_type_ids, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data

        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                             token_type_ids=token_type_ids.to(device))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        nsp_relationship_scores = model_output.nsp_relationship_scores.to("cpu")
        qa_start_logits = model_output.qa_start_logits.to("cpu")
        qa_end_logits = model_output.qa_end_logits.to("cpu")

        '''
        loss的计算
        '''
        loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100

        '''
        nsp loss 计算
        '''
        nsp_loss = loss_fct(nsp_relationship_scores.view(-1, 2), nsp_labels.view(-1))

        '''
        qa loss 计算
        '''
        try:
            start_loss = loss_fct(qa_start_logits, start_positions_labels)
            end_loss = loss_fct(qa_end_logits, end_positions_labels)
            qa_loss = (start_loss + end_loss) / 2
        except Exception as e:
            print(tokenizer.convert_ids_to_tokens(mask_input_ids[0]))
            print(e)
            exit(0)


        total_loss = torch.sqrt(torch.exp(nsp_loss)) * qa_loss  # 目的是为了当nsp预测错了的时候 加大惩罚程度
        # total_loss = torch.sqrt(torch.exp(nsp_loss)) * qa_loss  # 目的是为了当nsp预测错了的时候 加大惩罚程度

        # 进行统计展示
        epoch_step += 1

        epoch_nsp_loss += nsp_loss.detach()
        epoch_qa_loss += qa_loss.detach()
        epoch_total_loss += total_loss.detach()

        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        print('第%d个epoch的%d批数据的loss：%f' % (epoch + 1, step + 1, total_loss))
        total_loss.backward()  # 反向传播

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
        scheduler.step()  # warm_up
    # numpy不可以直接在有梯度的数据上获取，需要先去除梯度
    # 绘制epoch以及对应的测试集损失loss 第一个参数是y  第二个是x
    viz.line(Y=[(epoch_nsp_loss / epoch_step, epoch_qa_loss / epoch_step,
                 epoch_total_loss / epoch_step)],
             X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_1", update='append')

    # 绘制评估函数相关数据
    evaluate(model=model, eval_data_loader=dev_dataloader, epoch=epoch, tokenizer=tokenizer)

    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), 'save_model/nsp_qa_lstm/nsp_qa_lstm_epoch_%d' % (epoch + 1))

# 最后保存一下
torch.save(model.state_dict(), 'save_model/nsp_qa_lstm/nsp_qa_lstm_epoch_%d' % (epoch_size))
