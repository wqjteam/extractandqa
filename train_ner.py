import sys
from functools import partial

import pandas as pd
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import torch.utils.data as Data
from transformers.data.data_collator import tolist
from visdom import Visdom

from PraticeOfTransformers import CustomModelForNer, Utils
from PraticeOfTransformers.CustomModelForNSPQA import BertForUnionNspAndQA
from datasets import load_dataset

from PraticeOfTransformers.CustomModelForNer import BertForNerAppendBiLstmAndCrf

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if len(sys.argv) >= 2:
    batch_size = int(sys.argv[1])
if len(sys.argv) >= 3:
    epoch_size = int(sys.argv[2])
#获取模型路径
if len(sys.argv) >= 4 :
    model_name=sys.argv[3]

ner_id_label = {0: 'O', 1: 'B-ORG', 2: 'M-ORG', 3: 'E-ORG', 4: 'B-LOC', 5: 'M-LOC', 6: 'E-LOC', 7: 'B-PER',
                8: 'M-PER', 9: 'E-PER', 10: 'B-Time', 11: 'M-Time', 12: 'E-Time', 13: 'B-Book', 14: 'M-Book',
                15: 'E-Book',16:'I-ORG',17:'I-LOC',18:'I-PER',19:'I-Time',20:'I-Book'}
ner_label_id = {}
for key in ner_id_label:
    ner_label_id[ner_id_label[key]] = key
model = BertForNerAppendBiLstmAndCrf.from_pretrained(pretrained_model_name_or_path=model_name,
                                                     num_labels=len(ner_label_id))  # num_labels 测试用一下，看看参数是否传递
batch_size = 4
epoch_size = 10
# 用于梯度回归
optim = AdamW(model.parameters(), lr=5e-5)  # 需要填写模型的参数

print(model)


'''
获取数据
'''

# 加载conll2003数据集

nerdataset = Utils.convert_ner_data('data/origin/intercontest/project-1-at-2023-02-13-15-23-af00a0ee.json')
train_data, dev_data = Data.random_split(nerdataset, [int(len(nerdataset) * 0.9),
                                                      len(nerdataset) - int(
                                                          len(nerdataset) * 0.9)])

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
    tokenized_inputs = tokenizer(tokens, add_special_tokens=False, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(takens_labels):
        # 获取subtokens位置
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        # 遍历subtokens位置索引
        for word_idx in word_ids:
            if word_idx is None:
                # 将特殊字符的label设置为-100
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-----------------------------------训练模式为%s------------------------------------" % device)
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
    dev_data, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)




# 实例化相关metrics的计算对象
model_recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=len(ner_id_label)).to(device)
model_precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=len(ner_id_label)).to(device)
model_f1 = torchmetrics.F1Score(task='multiclass', average="macro", num_classes=len(ner_id_label)).to(device)

viz = Visdom(env=u'ner_%s_train'%(model_name))
name = ['total_loss']
name_precision_recall_f1 = ['precision_score','recall_score', 'f1_score']
viz.line(Y=[(0.)], X=[(0.)], win="pitcure_1",
         opts=dict(title='train_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置 #win指的是图形id
viz.line(Y=[(0.)], X=[(0.)], win="pitcure_2",
         opts=dict(title='eval_loss', legend=name, xlabel='epoch', ylabel='loss', markers=False))  # 绘制起始位置
viz.line(Y=[(0., 0.)], X=[(0., 0.)], win="pitcure_3",
         opts=dict(title='eval_precision_recall_f1', legend=name_precision_recall_f1, xlabel='epoch', ylabel='score',
                   markers=False))  # 绘制起始位置







#  评估函数，用作训练一轮，评估一轮使用
def evaluate(model, eval_data_loader, epoch):
    #评估之前将其重置
    eval_total_loss = 0
    eval_step = 0
    # 依次处理每批数据
    for return_batch_data in eval_data_loader:  # 一个batch一个bach的训练完所有数据

        input_ids = return_batch_data['input_ids']
        token_type_ids = return_batch_data['token_type_ids']
        attention_masks = return_batch_data['attention_mask']
        labels = return_batch_data['labels']

        model_output = model(input_ids.to(device), token_type_ids=token_type_ids.to(device), labels=labels.to(device))
        config = model.config
        loss, outputs = model_output
        predict=torch.argmax(outputs,dim=2)

        '''
        update 是计算当个batch的值  compute计算所有累加的值
        '''
        precision_score=model_precision(predict, labels.to(device))
        model_precision.update(predict, labels.to(device))
        recall_score=model_recall(predict, labels.to(device))
        model_recall.update(predict, labels.to(device))
        f1_score=model_f1(predict, labels.to(device))
        model_f1.update(predict, labels.to(device))


        # 损失函数的平均值
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        # 进行统计展示
        eval_step += 1


        eval_total_loss += loss.detach().to('cpu')

        print('--eval---eopch: %d --precision得分: %.6f--recall得分: %.6f--- f1得分: %.6f- 损失函数: %.6f' % ( epoch, precision_score, recall_score,f1_score, total_loss))
    viz.line(Y=[eval_total_loss / eval_step], X=[epoch + 1], win="pitcure_2", update='append')
    viz.line(Y=[(model_precision.compute().to('cpu'), model_recall.compute().to('cpu') ,model_f1.compute().to('cpu'))],
             X=[(epoch + 1, epoch + 1, epoch + 1)], win="pitcure_3", update='append')
    model_precision.reset()
    model_recall.reset()
    model_f1.reset()











for epoch in range(epoch_size):  # 所有数据迭代总的次数

    total_loss=0
    total_step=0
    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据

        input_ids = return_batch_data['input_ids']
        token_type_ids = return_batch_data['token_type_ids']
        attention_masks = return_batch_data['attention_mask']
        labels = return_batch_data['labels']

        model_output = model(input_ids.to(device), token_type_ids=token_type_ids.to(device), labels=labels.to(device))
        config = model.config
        loss, outputs = model_output
        optim.zero_grad()  # 每次计算的时候需要把上次计算的梯度设置为0

        total_step+=1
        total_loss+=loss.detach().cpu()


        print('第%d个epoch的%d批数据的loss：%f' % (epoch + 1, step + 1, loss))
        viz.line(Y=[total_loss / total_step],X=[ epoch + 1], win="pitcure_1", update='append')

        evaluate(model, dev_dataloader, epoch)
        loss.backward()  # 反向传播
        optim.step()  # 用来更新参数，也就是的w和b的参数更新操作