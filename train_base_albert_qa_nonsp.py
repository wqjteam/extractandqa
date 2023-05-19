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
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModel, \
    AutoModelForQuestionAnswering, ErnieForQuestionAnswering, AlbertForQuestionAnswering, BertTokenizerFast
from visdom import Visdom

import CommonUtil
import data_get_qa_all_label
from PraticeOfTransformers import Utils
from PraticeOfTransformers.CustomModelForNSPQABILSTM import CustomModelForNSPQABILSTM

model_name = 'voidful/albert_chinese_base'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

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

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

'''
获取数据
'''
origin_data = pd.read_json("./data/origin/intercontest/culture_qa_error_postivate_train.json",
                           orient='records',
                           lines=True)


# train_data = train_data[train_data['nsp'].apply(lambda x: x == 1)]
#
# train_data = train_data[train_data['passage'].apply(lambda x: x.startswith('J Storm'))]


# passage_keyword_json = passage_keyword_json[:10]
# print(111)


def tokenize_and_align_labels(data, tokenizer):
    text, q_a, nsp = zip(*data)  # arrat的四列 转为tuple
    textarray = []

    # text = [list(data) for data in text]  # tuple 转为 list0
    questions = [list(qa.get('question')) for qa in q_a]
    answers = [qa.get('answer').get('text') for qa in q_a]
    token_answers_index = [(qa.get('answer').get('start'), qa.get('answer').get('end')) for qa in q_a]
    nsps = list(nsp)  # tuple 转为list

    """
    处理大小写、空格与越界导致的对不齐
    """
    for index, data in enumerate(text):
        start_index, end_index = token_answers_index[index]
        start_decrease = 0
        end_decrease = 0
        for i, x in enumerate(data):
            if x == '' or x == ' ':
                if i <= start_index:
                    start_decrease += 1
                if i <= end_index:
                    end_decrease += 1
        token_answers_index[index] = (start_index - start_decrease, end_index - end_decrease)
        data = [x.lower() for x in data if x != '' and x != ' ']

        if len(data) > 512 - 3 - 30:  # 给问题留30个字
            textarray.append(list(data[:512 - 3 - 30]))
        else:
            textarray.append(list(data))

    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置 需要在进行encode之后再进行添加
    end_positions_labels = []  # 记录终止始位置 需要在进行encode之后再进行添加

    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''
    # tokenizer.tokenize(textarray[0])
    encoded_dict_textandquestion = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(textarray, questions)),  # 输入文本对 # 输入文本,采用list[tuple(text,question)]的方式进行输入
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=512,  # 填充 & 截断长度
        truncation=True,
        padding='longest',
        return_attention_mask=True,  # 返回 attn. masks.
        is_split_into_words=True
    )
    # print(len(textarray[0]))
    # print(textarray[0][:10])
    # print(len(textarray[0])+len(questions[0]))
    # print(len(encoded_dict_textandquestion['input_ids'][0]))
    for index in range(len(textarray)):
        # 获取subtokens位置
        passage_len = len(textarray[index])
        back_start_index, back_end_index = token_answers_index[index]
        start_index, end_index = token_answers_index[index]
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
            stpid=tokenizer.convert_tokens_to_ids('[SEP]')
            seqindex = encoded_dict_textandquestion['input_ids'][index].index(stpid)  # ['SEP']
            token_answers_index[index] = (seqindex, seqindex)
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
                token_answers_index[index] = (true_start_index, true_end_index)
                currenttext = ''.join(textarray[index])

                currenttexttoken = ''.join(tokenizer.convert_ids_to_tokens(
                    encoded_dict_textandquestion['input_ids'][index]))
                currentnsp = nsps[index]
                currentquestion = questions[index]
                currentanswer = answers[index]
                currentanswertext = ''.join(textarray[index][back_start_index: back_end_index])
                currenttokennanswer = ''.join(tokenizer.convert_ids_to_tokens(
                    encoded_dict_textandquestion['input_ids'][index][true_start_index: true_end_index]))
                break
            previous_word_idx = word_idx
    returndata = []
    for rdata in zip(textarray, encoded_dict_textandquestion['input_ids'],
                     encoded_dict_textandquestion['attention_mask']
            , encoded_dict_textandquestion['token_type_ids'], questions, answers,
                     token_answers_index, nsps):
        returndata.append(list(rdata))
    return returndata


origin_data = tokenize_and_align_labels(origin_data.explode("q_a").values, tokenizer)

train_data, dev_data = Data.random_split(origin_data, [int(len(origin_data) * 0.9),
                                                       len(origin_data) - int(
                                                           len(origin_data) * 0.9)],
                                         generator=torch.Generator().manual_seed(0))
# train_data = train_data.dataset[0:2]
# dev_data = dev_data[0:2]
# test_data = test_data[0:2]
nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}

# print(output)
# 用于梯度回归

optim = AdamW(model.parameters(), lr=learning_rate)
train_steps_per_epoch = len(list(train_data)) // batch_size
scheduler = get_cosine_schedule_with_warmup(optim,
                                            num_warmup_steps=(epoch_size // 10) * train_steps_per_epoch,
                                            num_training_steps=epoch_size * train_steps_per_epoch)


def create_batch(data, tokenizer):
    text, input_ids, attention_mask, token_type_ids, questions, answers, answers_index, nsps = zip(
        *data)  # arrat的四列 转为tuple

    # mask_input_labels 位置用torch.tensor(encoded_dict_textandquestion['input_ids'])代替了，这里不计算mlm的loss了
    return torch.tensor(list(input_ids)), torch.tensor(
        list(attention_mask)), torch.tensor(list(token_type_ids)), torch.tensor(list(nsps)), torch.tensor(
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

# 看是否用cpu或者gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)

if torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))

    model = nn.DataParallel(model, device_ids=device_ids)

model.to(device)
model_name = model_name.split("/")[1]
'''
可视化
'''
viz = Visdom(env=u'qa_%s_train' % (model_name))
name = ['nsp_loss', 'qa_loss', 'total_loss']
name_em_f1 = ['em_score', 'f1_score']
viz.line(Y=[0.], X=[0.], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[0.], X=[0.], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置
viz.line(Y=[(0., 0.)], X=[(0., 0.)], win="pitcure_3",
         opts=dict(title='eval_em_f1', legend=name_em_f1, xlabel='epoch', ylabel='score(100分制)',
                   markers=False))  # 绘制起始位置
print("--------------------Visdom已完成注册---------------")


#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch, tokenizer):
    eval_total_loss = 0
    eval_em_score = 0
    eval_f1_score = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据
        mask_input_ids, attention_masks, token_type_ids, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data
        with torch.no_grad():
            model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                                 token_type_ids=token_type_ids.to(device),
                                 start_positions=start_positions_labels.to(device),
                                 end_positions=end_positions_labels.to(device))

        # 进行转换
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        loss = model_output.loss.to("cpu")
        qa_start_logits = model_output.start_logits.to("cpu")
        qa_end_logits = model_output.end_logits.to("cpu")
        eval_total_loss += loss
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
        eval_em_score += em_score
        eval_f1_score += f1_score

        print('--eval---epoch次数:%d---em得分: %.6f - f1得分: %.6f-- total损失函数: %.6f'
              % (epoch + 1, em_score, f1_score, eval_total_loss.detach()))
        # 损失函数的平均值
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        # 进行统计展示
        eval_step += 1

    viz.line(Y=[eval_total_loss / eval_step], X=[epoch + 1], win="pitcure_2", update='append')
    viz.line(Y=[(eval_em_score / eval_step, eval_f1_score / eval_step)],
             X=[(epoch + 1, epoch + 1)], win="pitcure_3", update='append')
    if eval_em_score / eval_step >= 100.70:
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
        qa_start_logits = model_output.start_logits.to("cpu")
        qa_end_logits = model_output.end_logits.to("cpu")
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
        torch.save(model.state_dict(), 'save_model/qa_%s/ultimate_qa_epoch_%d' % (epoch_size))
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
                             token_type_ids=token_type_ids.to(device),
                             start_positions=start_positions_labels.to(device),
                             end_positions=end_positions_labels.to(device))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config

        loss = model_output.loss.to("cpu")
        qa_start_logits = model_output.start_logits.to("cpu")
        qa_end_logits = model_output.end_logits.to("cpu")
        epoch_nsp_loss += loss

        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        print('第%d个epoch的%d批数据的loss：%f' % (epoch + 1, step + 1, loss))
        loss.backward()  # 反向传播

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
        scheduler.step()  # warm_up
    # numpy不可以直接在有梯度的数据上获取，需要先去除梯度
    # 绘制epoch以及对应的测试集损失loss 第一个参数是y  第二个是x
    viz.line(Y=[epoch_nsp_loss.detach() / epoch_step + 1], X=[epoch + 1], win="pitcure_1", update='append')

    # 绘制评估函数相关数据
    evaluate(model=model, eval_data_loader=dev_dataloader, epoch=epoch, tokenizer=tokenizer)

    # 每5个epoch保存一次
    # if (epoch + 1) % 5 == 0:
    #     torch.save(model.state_dict(), 'save_model/qa_%s/qa_epoch_%d' % (model_name, epoch + 1))

# 最后保存一下
# torch.save(model.state_dict(), 'save_model/qa_%s/qa_epoch_all_epoch_%d' % (model_name, epoch_size))
