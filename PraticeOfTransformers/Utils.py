import json

import collections
import os

import pandas as pd
from tqdm import tqdm


def pad_sequense_python(list_args, fillvalue):
    my_len = [len(k) for k in list_args]
    max_num = max(my_len)
    result = []

    for my_list in list_args:
        if len(my_list) < max_num:
            for i in range(max_num - len(my_list)):
                my_list.append(fillvalue)

        result.append(my_list)

    return result


def get_eval(pred_arr, target_arr): #pred_arr与target_arr 必须是二维数组

    total_f1 = 0
    total_em = 0
    # 如果两个pred 和target数量都不相等
    if len(pred_arr) != len(target_arr):
        pass
    else:
        for pred_array, real_array in zip(pred_arr, target_arr):
            f1 = compute_f1(a_gold=real_array, a_pred=pred_array)
            em = compute_exact(a_gold=real_array, a_pred=pred_array)
            total_em += em
            total_f1 += f1

    metric = {'EM': (total_em / len(target_arr)) * 100,
              'F1': (total_f1 / len(target_arr)) * 100}
    return metric


## 去除标点及小写 去除特殊字符
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace"""
    punctuation = r"""!"#$%&'()*+_,./:;<>=?@[]\^_`{}|~！￥……（）——【】’：；，《》“。，、？"""


    def remove_punc(str_array):
        exclude = set(punctuation)
        removed_punc=[]
        #isinstance(ch, list)  做了改进，对几个特殊字符也放过了
        for ch in str_array:
            if  isinstance(ch, list) or ch not in exclude:
                removed_punc.append(ch)
            else:
                pass
        return removed_punc



    return remove_punc(s)


def get_token(s):
    return [c for c in normalize_answer(s)]


## 计算 f1
'''
F1:计算预测出的答案与原始答案字符之间的overlap，
根据overlap的数目与原始ground truth answer的字符数目计算回召率，
overlap的数目与预测出的所有字符数目计算准确率
'''
def compute_f1(a_gold, a_pred):
    gold_toks = get_token(a_gold)
    pred_toks = get_token(a_pred)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())  #共同出现的词的个数
    #精确率(precision)是 指预测的答案有多大比例的单词在标准答案中出现
    precision = 1.0 * num_same / len(gold_toks)

    #召回率(recall)是 指标准答案中的单词有多大比例在预测答案中出现
    recall = 1.0 * num_same / len(pred_toks)
    if precision + recall==0: #如果分母为0 则直接返回为0
        return 0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


## 计算 em
'''
EM：表示完全匹配的，如果完全匹配则为1，否则为0
'''


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))



'''
处理subword的情况，并把英文小写
'''
def get_all_word(tokenizer,bacth_id):
    batch_whole_word_array=[]
    if bacth_id is None:  # 判断一下是否为空
        return batch_whole_word_array
    for id in bacth_id:

        subword = tokenizer.convert_ids_to_tokens(id)


        if subword.startswith('##'):
            #有时间预测有问题，预测到subword开头了
            if len(batch_whole_word_array)==0:
                batch_whole_word_array.append(subword[2:])
            else:
                batch_whole_word_array[-1]=batch_whole_word_array[-1]+subword[2:]
        else:
            batch_whole_word_array.append(subword)

        tempstr = batch_whole_word_array[-1] #如果有英文，把英文小写
        if (u'\u0041' <= tempstr <= u'\u005a') or (u'\u0061' <= tempstr <= u'\u007a'):
            batch_whole_word_array[-1] = tempstr.lower()

    return batch_whole_word_array

def read(self, data_path):
    data_parts = ['train', 'valid', 'test']
    extension = '.txt'
    dataset = {}
    for data_part in tqdm(data_parts):
        file_path = os.path.join(data_path, data_part+extension)
        dataset[data_part] = self.read_file(str(file_path))
    return dataset

def read_file(self, file_path):
    samples = []
    tokens = []
    tags = []
    with open(file_path,'r', encoding='utf-8') as fb:
        for line in fb:
            line = line.strip('\n')

            if line == '-DOCSTART- -X- -X- O':
                # 去除数据头
                pass
            elif line =='':
                # 一句话结束
                if len(tokens) != 0:
                    samples.append((tokens, tags))
                    tokens = []
                    tags = []
            else:
                # 数据分割，只要开头的词和最后一个实体标注。
                contents = line.split(' ')
                tokens.append(contents[0])
                tags.append(contents[-1])
    return samples


#从json抓内text label的格式
def convert_ner_data(file_path):
    """
    file_path: 通过Label Studio导出的csv文件
    save_path: 保存的路径
    """
    data = pd.read_json(file_path)

    text_label_tuple=[]
    for idx, item in data.iterrows():
        text = item['text']
        if text is None:
            text = ''
        text_list = list(text)
        label_list = []
        labels = item['label']
        label_list = ['O' for i in range(len(text_list))] #默认添加
        if labels is  None :
            pass
        else:

            for label_item in labels:
                start = label_item['start']
                end = label_item['end']
                label = label_item['labels'][0]
                label_list[start] = f'B-{label}'
                label_list[start+1:end-1] = [f'M-{label}' for i in range(end-start-2)]
                label_list[end - 1] = f'E-{label}'
        assert len(label_list) == len(text_list)
        text_label_tuple.append((text_list,label_list))

    return text_label_tuple

#转换json 到qa
def convert_qa_nsp_data(file_path):
    pass





if __name__ == '__main__':
    data=convert_ner_data('../data/origin/intercontest/project-1-at-2023-02-13-15-23-af00a0ee.json')
    print(data)