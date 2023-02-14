from functools import partial

import pandas as pd
import torch
import torchmetrics
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import torch.utils.data as Data
from transformers.data.data_collator import tolist

from PraticeOfTransformers import CustomModelForNer, Utils
from PraticeOfTransformers.CustomModelForNSPQA import BertForUnionNspAndQA
from datasets import load_dataset

from PraticeOfTransformers.CustomModelForNer import BertForNerAppendBiLstmAndCrf



model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

ner_id_label = {0: 'O', 1: 'B-ORG', 2: 'M-ORG', 3: 'E-ORG', 4: 'B-LOC', 5: 'M-LOC', 6: 'E-LOC', 7: 'B-PER',
                8: 'M-PER', 9: 'E-PER', 10: 'B-Time', 11: 'M-Time', 12: 'E-Time', 13: 'B-Book', 14: 'M-Book',
                15: 'E-Book'}
ner_label_id = {}
for key in ner_id_label:
    ner_label_id[ner_id_label[key]] = key
model = BertForNerAppendBiLstmAndCrf.from_pretrained(pretrained_model_name_or_path=model_name,
                                                     num_labels=len(ner_label_id))  # num_labels 测试用一下，看看参数是否传递
batch_size = 4
epoch_size = 10
# 用于梯度回归
optim = AdamW(model.parameters(), lr=5e-5)  # 需要填写模型的参数

# model = BertForUnionNspAndQA.from_pretrained(model_name)
print(model)
# data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
#                                                              mlm=True,
#                                                              mlm_probability=0.15,
#                                                              return_tensors="pt")


'''
获取数据
'''




# 加载conll2003数据集

nerdataset = Utils.convert_ner_data('data/origin/intercontest/project-1-at-2023-02-13-15-23-af00a0ee.json')
train_data, dev_data = Data.random_split(nerdataset, [int(len(nerdataset) * 0.9),
                                                                len(nerdataset) - int(
                                                                    len(nerdataset) * 0.9)])
task = 'ner'
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
    tokenized_inputs = tokenizer(tokens,  add_special_tokens=False,  truncation=True, is_split_into_words=True)

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
ner_align_data_collator = DataCollatorForTokenClassification(tokenizer)  # 他会对于一些label的空余的位置进行补齐 对于data_collator输入必须有labels属性
# mlm_align_data_collator = DataCollatorForTokenClassification(tokenizer)   # 他会对于一些label的空余的位置进行补齐 对于data_collator输入必须有labels属性

# 看是否用cpu或者gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-----------------------------------训练模式为%s------------------------------------" % device)
model.to(device)


def create_batch(data, tokenizer, ner_data_collator):
    tokenized_datasets = tokenize_and_align_labels(data, tokenizer)
    # 投喂进ner_data_collator的数据必须是数组
    tokenized_datasets_list = [{'input_ids': i, 'token_type_ids': t,'attention_mask':a, 'labels': l} for i, t, a, l in
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



for epoch in range(epoch_size):  # 所有数据迭代总的次数

    for step, return_batch_data in enumerate(train_dataloader):  # 一个batch一个bach的训练完所有数据

        input_ids = return_batch_data['input_ids']
        token_type_ids = return_batch_data['token_type_ids']
        attention_masks = return_batch_data['attention_mask']
        labels = return_batch_data['labels']

        model_output = model(input_ids.to(device),token_type_ids=token_type_ids.to(device), labels=labels.to(device))
        config = model.config
        prediction_scores = model_output.mlm_prediction_scores.to("cpu")
        nsp_relationship_scores = model_output.nsp_relationship_scores.to("cpu")
        qa_start_logits = model_output.qa_start_logits.to("cpu")
        qa_end_logits = model_output.qa_end_logits.to("cpu")
