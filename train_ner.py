from functools import partial

import pandas as pd
import torch
import torchmetrics
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import torch.utils.data as Data

from PraticeOfTransformers import CustomModelForNer
from PraticeOfTransformers.CustomModelForNSPQA import BertForUnionNspAndQA
from datasets import load_dataset

from PraticeOfTransformers.CustomModelForNer import BertForNerAppendBiLstmAndCrf

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForNerAppendBiLstmAndCrf.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递
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
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

passage_keyword_json = passage_keyword_json.explode("q_a").values

train_data, dev_data = Data.random_split(passage_keyword_json, [int(len(passage_keyword_json) * 0.9),
                                                                len(passage_keyword_json) - int(
                                                                    len(passage_keyword_json) * 0.9)])

nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}

# 加载conll2003数据集

dataset = load_dataset("rotten_tomatoes", split="train")
dataset = load_dataset('conll2003')
example = dataset["train"][4]
task='ner'
label_all_tokens = True


'''
由于标注数据通常是在word级别进行标注的，既然word还会被切分成subtokens，那么意味着我们还需要对标注数据进行subtokens的对齐。
同时，由于预训练模型输入格式的要求，往往还需要加上一些特殊符号比如： [CLS] 和 [SEP]。
word_ids将每一个subtokens位置都对应了一个word的下标。
比如第1个位置对应第0个word，然后第2、3个位置对应第1个word。特殊字符对应了None。
有了这个list，我们就能将subtokens和words还有标注的labels对齐啦。
'''
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
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
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        # 对齐word
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True,remove_columns=datasets["train"].column_names,).data
# 数据收集器，用于将处理好的数据输入给模型
data_collator = DataCollatorForTokenClassification(tokenizer)   # 他会对于一些label的空余的位置进行补齐 对于data_collator输入必须有labels属性
a=data_collator(tokenized_datasets)

def create_batch(data, tokenizer, data_collator):
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








    base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict_textandquestion['input_ids']]
    attention_masks = [torch.tensor(attention) for attention in encoded_dict_textandquestion['attention_mask']]
    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask
    data_collator_output = data_collator()
    mask_input_ids = data_collator_output["input_ids"]

    mask_input_labels = data_collator_output["labels"]  # 需要获取不是-100的位置，证明其未被替换，这也是target -100的位置在计算crossentropyloss 会丢弃

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