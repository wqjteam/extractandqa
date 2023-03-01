# -*- coding: utf-8 -*-
import os
import sys
from functools import partial

import torch
import torch.utils.data as Data
import torchmetrics
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorForTokenClassification, get_cosine_schedule_with_warmup, AutoConfig
from visdom import Visdom

from PraticeOfTransformers import Utils
from PraticeOfTransformers.CustomModelForNer import BertForNerAppendBiLstmAndCrf

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 指定GPU编号 多gpu训练
print('--------------------sys.argv:%s-------------------' % (','.join(sys.argv)))

model_name = 'bert-base-chinese'
batch_size = 2
epoch_size = 500
learning_rate = 1e-5
weight_decay = 0.01  # 最终目的是防止过拟合
full_fine_tuning = True
tokenizer = AutoTokenizer.from_pretrained(model_name)
if len(sys.argv) >= 2:
    batch_size = int(sys.argv[1])
if len(sys.argv) >= 3:
    epoch_size = int(sys.argv[2])

ner_id_label = {0: '[CLS]', 1: '[SEP]', 2: 'O', 3: 'B-ORG', 4: 'B-PER', 5: 'B-LOC', 6: 'B-TIME', 7: 'B-BOOK',
                8: 'I-ORG', 9: 'I-PER', 10: 'I-LOC', 11: 'I-TIME', 12: 'I-BOOK'}
ner_label_id = {}
for key in ner_id_label:
    ner_label_id[ner_id_label[key]] = key

model = BertForNerAppendBiLstmAndCrf.from_pretrained(pretrained_model_name_or_path=model_name,
                                                     num_labels=len(ner_label_id))  # num_labels 测试用一下，看看参数是否传递
# 获取模型路径
if len(sys.argv) >= 4:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=len(ner_label_id))
    model = BertForNerAppendBiLstmAndCrf(config)
    # 因为后面的参数没有初始化，所以采用非强制性约束
    model.load_state_dict(torch.load(sys.argv[3]), strict=False)
    model_name = "special-keyword-bert-chinese"

# 加载数据集
nerdataset = Utils.convert_ner_data('data/origin/intercontest/relic_ner_handlewell.json')
# nerdataset = list(filter(lambda x: ''.join(x[0]).startswith("小双桥遗址"), nerdataset))
# nerdataset=nerdataset[0:100]
train_data, dev_data = Data.random_split(nerdataset, [int(len(nerdataset) * 0.9),
                                                      len(nerdataset) - int(
                                                          len(nerdataset) * 0.9)],
                                         generator=torch.Generator().manual_seed(0))

'''
非bert层的学习率需要提高 crf需要为bert的
bert 给是10−5量级
crf  给10−2量级别 即bert的是1000倍
而对其biases参数和BN层的gamma和beta参数不进行衰减
首先正则化主要是为了防止过拟合，而过拟合一般表现为模型对于输入的微小改变产生了输出的较大差异，
这主要是由于有些参数w过大的关系，通过对||w||进行惩罚，可以缓解这种问题。
而如果对||b||进行惩罚，其实是没有作用的，因为在对输出结果的贡献中
参数b对于输入的改变是不敏感的，不管输入改变是大还是小，参数b的贡献就只是加个偏置而已，他不背锅
'''
# 用于梯度回归
if full_fine_tuning:
    # model.named_parameters(): [bert, bilstm, classifier, crf]
    bert_optimizer = list(model.bert.named_parameters())
    lstm_optimizer = list(model.bilstm.named_parameters())
    classifier_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],  # 对于在no_decay 不进行正则化
         'weight_decay': 0.0},
        {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 100, 'weight_decay': weight_decay},
        {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
         'lr': learning_rate * 100, 'weight_decay': 0.0},
        {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
         'lr': learning_rate * 100, 'weight_decay': weight_decay},
        {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
         'lr': learning_rate * 100, 'weight_decay': 0.0},
        {'params': model.crf.parameters(), 'lr': learning_rate * 1000}
    ]
    # only fine-tune the head classifier
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
train_steps_per_epoch = len(train_data) // batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(epoch_size // 10) * train_steps_per_epoch,
                                            num_training_steps=epoch_size * train_steps_per_epoch)
optim = AdamW(model.parameters(), lr=learning_rate)  # 需要填写模型的参数

print(model)

'''
获取数据
'''

label_all_tokens = True

'''
由于标注数据通常是在word级别进行标注的，既然word还会被切分成subtokens，那么意味着我们还需要对标注数据进行subtokens的对齐。
同时，由于预训练模型输入格式的要求，往往还需要加上一些特殊符号比如： [CLS] 和 [SEP]。
word_ids将每一个subtokens位置都对应了一个word的下标。
比如第1个位置对应第0个word，然后第2、3个位置对应第1个word。特殊字符对应了None。
有了这个list，我们就能将subtokens和words还有标注的labels对齐啦。
'''


def tokenize_and_align_labels(examples, tokenizer):
    tokens, takens_labels = zip(*examples)
    # tokens, takens_labels = examples
    tokenized_inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=list(tokens), add_special_tokens=True,
                                                   truncation=True, is_split_into_words=True)
    # print(''.join(tokens[0]))
    # print(''.join(tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])))
    labels = []
    for i, label in enumerate(takens_labels):
        # 获取subtokens位置
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        # 遍历subtokens位置索引
        for word_index, word_idx in enumerate(word_ids):
            # 处理特殊字符
            if word_idx is None:

                if word_index == 0:
                    label_ids.append(ner_label_id['[CLS]'])
                elif word_index == len(word_ids) - 1:
                    label_ids.append(ner_label_id['[SEP]'])
                else:
                    # 将特殊字符的label设置为-100,不会出现 pad的，但还是设置下
                    label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(ner_label_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(ner_label_id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        # 对齐word
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 数据收集器，用于将处理好的数据输入给模型
ner_align_data_collator = DataCollatorForTokenClassification(
    tokenizer)  # 他会对于一些label的空余的位置进行补齐 对于data_collator输入必须有labels属性
# mlm_align_data_collator = DataCollatorForTokenClassification(tokenizer)   # 他会对于一些label的空余的位置进行补齐 对于data_collator输入必须有labels属性

# 看是否用cpu或者gpu训练
# 看是否用cpu或者gpu训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-----------------------------------训练模式为%s------------------------------------" % device)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))

    model = nn.DataParallel(model, device_ids=device_ids)

model.to(device)
model.train()


def create_batch(data, tokenizer, ner_data_collator):
    tokenized_datasets = tokenize_and_align_labels(data, tokenizer)
    # 投喂进ner_data_collator的数据必须是数组
    tokenized_datasets_list = [{'input_ids': i, 'token_type_ids': t, 'attention_mask': a, 'labels': l} for i, t, a, l in
                               zip(tokenized_datasets['input_ids'], tokenized_datasets['token_type_ids'],
                                   tokenized_datasets['attention_mask'], tokenized_datasets['labels'])]
    ner_data_collator_data = ner_data_collator(tokenized_datasets_list)
    # 对于model只接受tensor[list] 必须把 list[tensor] 转为tensor[list]
    return ner_data_collator_data


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer, ner_data_collator=ner_align_data_collator)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    train_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)

dev_dataloader = Data.DataLoader(
    dev_data, shuffle=False, collate_fn=create_batch_partial, batch_size=batch_size
)

# 实例化相关metrics的计算对象   len(ner_id_label)+1是为了ignore_index用 他不允许为负数值 这里忽略了，所以不影响结果
model_recall = torchmetrics.Recall(num_classes=len(ner_id_label) + 1, mdmc_average='global',
                                   ignore_index=len(ner_id_label)).to(device)
model_precision = torchmetrics.Precision(num_classes=len(ner_id_label) + 1, mdmc_average='global',
                                         ignore_index=len(ner_id_label)).to(device)
model_f1 = torchmetrics.F1Score(num_classes=len(ner_id_label) + 1, mdmc_average='global',
                                ignore_index=len(ner_id_label)).to(device)

viz = Visdom(env=u'ner_%s_train' % (model_name))
name = ['total_loss']
name_precision_recall_f1 = ['precision_score', 'recall_score', 'f1_score']
viz.line(Y=[(0.)], X=[(0.)], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[(0.)], X=[(0.)], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置
viz.line(Y=[(0., 0., 0.)], X=[(0., 0., 0.)], win="pitcure_3",
         opts=dict(title='eval_precision_recall_f1', legend=name_precision_recall_f1, xlabel='epoch', ylabel='score',
                   markers=False))  # 绘制起始位置


#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch):
    # 评估之前将其重置
    eval_total_loss = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据

        input_ids = return_batch_data['input_ids']
        token_type_ids = return_batch_data['token_type_ids']
        attention_masks = return_batch_data['attention_mask']
        labels = return_batch_data['labels']

        model_output = model(input_ids.to(device), token_type_ids=token_type_ids.to(device), labels=labels.to(device),
                             is_test=True)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config
        loss, outputs = model_output

        predict = outputs.view(-1, outputs.shape[2])

        # 这里方便计算用，-100 torchmetrics无法使用
        # predict[predict == -100] = len(ner_id_label)
        labels[labels == -100] = len(ner_id_label)  # predict 或者 labels 只要出现ignore_index 则不会统计在内
        '''
        update 是计算当个batch的值  compute计算所有累加的值
        '''
        precision_score = model_precision(predict, labels.to(device))
        model_precision.update(predict, labels.to(device))
        recall_score = model_recall(predict, labels.to(device))
        model_recall.update(predict, labels.to(device))
        f1_score = model_f1(predict, labels.to(device))
        model_f1.update(predict, labels.to(device))

        # 损失函数的平均值
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        # 进行统计展示
        eval_step += 1

        eval_total_loss += torch.mean(loss).detach().cpu()

        print('--eval---eopch: %d --precision得分: %.6f--recall得分: %.6f--- f1得分: %.6f- 损失函数: %.6f' % (
            epoch, precision_score, recall_score, f1_score, torch.mean(loss).detach().cpu()))
    viz.line(Y=[eval_total_loss / eval_step], X=[epoch + 1], win="pitcure_2", update='append')
    viz.line(Y=[(model_precision.compute().to('cpu'), model_recall.compute().to('cpu'), model_f1.compute().to('cpu'))],
             X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_3", update='append')
    model_precision.reset()
    model_recall.reset()
    model_f1.reset()


for epoch in range(epoch_size):  # 所有数据迭代总的次数

    total_loss = 0
    total_step = 0
    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据

        input_ids = return_batch_data['input_ids']
        token_type_ids = return_batch_data['token_type_ids']
        attention_masks = return_batch_data['attention_mask']
        labels = return_batch_data['labels']

        model_output = model(input_ids.to(device), token_type_ids=token_type_ids.to(device), labels=labels.to(device))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model_config = model.module.config
        else:
            model_config = model.config
        loss, outputs = model_output
        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        total_step += 1
        total_loss += torch.mean(loss).detach().cpu()

        print('第%d个epoch的%d批数据的loss：%f' % (epoch + 1, step + 1, torch.mean(loss).detach().cpu()))

        loss.backward(torch.ones_like(loss))  # 反向传播
        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作
        scheduler.step()  # warm_up
    viz.line(Y=[total_loss / total_step], X=[epoch + 1], win="pitcure_1", update='append')
    evaluate(model, dev_dataloader, epoch)
    # 每5个epoch保存一次
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), 'save_model/ner/ner_epoch_%d' % (epoch + 1))
# 最后保存一下
torch.save(model.state_dict(), 'save_model/ner/ner_epoch_%d' % (epoch_size))
