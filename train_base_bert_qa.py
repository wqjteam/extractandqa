# -*- coding: utf-8 -*-
import os
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
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, AutoModelForQuestionAnswering

import CommonUtil
from PraticeOfTransformers import Utils
from PraticeOfTransformers.DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial
from PraticeOfTransformers.CustomModelForNSPQA import BertForUnionNspAndQA, NspAndQAConfig
from PraticeOfTransformers.DataCollatorForWholeWordMaskOriginal import DataCollatorForWholeWordMaskOriginal
from PraticeOfTransformers.DataCollatorForWholeWordMaskSpecial import DataCollatorForWholeWordMaskSpecial
import sys

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' #指定GPU编号 多gpu训练
batch_size = 2
epoch_size = 1000

print('--------------------sys.argv:%s-------------------' % (','.join(sys.argv)))
if len(sys.argv) >= 2:
    batch_size = int(sys.argv[1])
if len(sys.argv) >= 3:
    epoch_size = int(sys.argv[2])
data_collator = DataCollatorForWholeWordMaskSpecial(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.15,
                                                    return_tensors="pt")


keyword_flag = False



model = AutoModelForQuestionAnswering.from_pretrained(model_name)  # num_labels 测试用一下，看看参数是否传递



# 用于梯度回归
optim = Adam(model.parameters(), lr=5e-5)  # 需要填写模型的参数
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


train_data, dev_data = Data.random_split(passage_keyword_json, [int(len(passage_keyword_json) * 0.9),
                                                                len(passage_keyword_json) - int(
                                                                 len(passage_keyword_json) * 0.9)])




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

dev_dataloader = Data.DataLoader(
    dev_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)


# 看是否用cpu或者gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)

if torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))

    model = nn.DataParallel(model,device_ids=device_ids)

model.to(device)

model_name='base_bert_model'
'''
可视化
'''
viz = Visdom(env=u'qa_base_bert_model_notnsp')
name = [ 'total_loss']
name_em_f1 = ['em_score', 'f1_score']
viz.line(Y=[ 0.], X=[ 0.], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[ 0.], X=[ 0.], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置
viz.line(Y=[(0., 0.)], X=[(0., 0.)], win="pitcure_3",
         opts=dict(title='eval_em_f1', legend=name_em_f1, xlabel='epoch', ylabel='score(100分制)',
                   markers=False))  # 绘制起始位置



#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch, tokenizer):

    eval_total_loss = 0
    eval_em_score = 0
    eval_f1_score = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据
        mask_input_ids, attention_masks, token_type_ids, mask_input_labels, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data
        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),start_positions=start_positions_labels.to(device),end_positions=end_positions_labels.to(device),
                             token_type_ids=token_type_ids.to(device),return_dict=True)

        # 进行转换
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config=model.module.config
        else:
            model_config = model.config
        loss = model_output.loss
        qa_start_logits = model_output.start_logits.to("cpu")
        qa_end_logits = model_output.end_logits.to("cpu")

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

        print('--eval---epoch次数:%d---em得分: %.6f - f1得分: %.6f---total损失函数: %.6f'
              %(epoch ,em_score, f1_score, loss.detach()))
        # 损失函数的平均值
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        # 进行统计展示
        eval_step += 1
        eval_total_loss += loss.detach()
        eval_em_score += em_score
        eval_f1_score += f1_score

    viz.line(Y=[eval_total_loss / eval_step],X=[epoch + 1], win="pitcure_2", update='append')
    viz.line(Y=[(eval_em_score / eval_step, eval_f1_score / eval_step)],
             X=[(epoch + 1, epoch + 1)], win="pitcure_3", update='append')





# 进行训练
for epoch in range(epoch_size):  # 所有数据迭代总的次数
    epoch_mlm_loss = 0
    epoch_nsp_loss = 0
    epoch_qa_loss = 0
    epoch_total_loss = 0
    epoch_step = 0
    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据

        mask_input_ids, attention_masks, token_type_ids, mask_input_labels, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data

        model_output = model(input_ids=mask_input_ids.to(device), attention_mask=attention_masks.to(device),
                             token_type_ids=token_type_ids.to(device),start_positions=start_positions_labels.to(device),
                             end_positions=end_positions_labels.to(device),return_dict=True)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config
        loss=model_output.loss
        qa_start_logits = model_output.start_logits.to("cpu")
        qa_end_logits = model_output.end_logits.to("cpu")
        epoch_total_loss+=loss.to("cpu")


        # 进行统计展示
        epoch_step += 1


        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0
        print(loss)
        print('第%d个epoch的%d批数据的loss：%f' % (epoch + 1, step + 1, loss.detach().to("cpu")))
        loss.backward()  # 反向传播

        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
    # numpy不可以直接在有梯度的数据上获取，需要先去除梯度
    # 绘制epoch以及对应的测试集损失loss 第一个参数是y  第二个是x
    viz.line(Y=[epoch_total_loss / epoch_step],X=[ epoch + 1], win="pitcure_1", update='append')

    # 绘制评估函数相关数据
    evaluate(model=model, eval_data_loader=dev_dataloader, epoch=epoch, tokenizer=tokenizer)

    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), 'save_model/nsp_qa_base_bert/nsp_qa_base_bert_epoch_%d' % (epoch + 1))

# 最后保存一下
torch.save(model.state_dict(), 'save_model/nsp_qa_base_bert/nsp_qa_base_bert_epoch_%d' % (epoch_size))
