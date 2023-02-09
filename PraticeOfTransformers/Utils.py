import json

import collection


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


def get_eval(pred, target):
    pred
    total_f1 = 0
    total_em = 0
    # 如果两个pred 和target数量都不相等
    if len(pred) != len(target):
        pass
    else:
        for pred_array, real_array in zip(pred, target):
            f1 = compute_f1(a_gold=real_array, a_pred=pred_array)
            em = compute_exact(a_gold=real_array, a_pred=pred_array)
            total_em += em
            total_f1 += f1

    metric = {'EM': (total_em / len(target)) * 100,
              'F1': (total_f1 / len(target)) * 100}
    return metric


## 去除标点及小写
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace"""
    punctuation = r"""!"#$%&'()*+_,./:;<>=?@[]\^_`{}|~！￥……（）——【】’：；，《》“。，、？"""

    def white_space_fix(str_array):
        return ' '.join(str_array.split())

    def remove_punc(str_array):
        exclude = set(punctuation)
        removed_punc=[]
        #isinstance(ch, list)  做了改进，对几个特殊字符也放过了
        for ch in str_array:
            if  isinstance(ch, list) or ch not in exclude:
                removed_punc.append(ch)
            else:
                pass
        return ''.join(removed_punc)



    return white_space_fix(remove_punc(s))


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

    common = collection.Counter(gold_toks) & collection.Counter(pred_toks)
    num_same = sum(common.values())

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)

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
def get_all_word(tokenizer,bact_id_or_idarray):
    batch_whole_word_array=[]
    if bact_id_or_idarray is None:  # 判断一下是否为空
        return batch_whole_word_array
    for id_or_idarray in bact_id_or_idarray:




        if isinstance(id_or_idarray, int): #判断是否为数据  #这种情况为数字
            batch_whole_word_array.append(tokenizer.convert_ids_to_tokens(id_or_idarray))
        else:  #这种情况为 subword  是个数组
            subwordconcat=''
            for id in id_or_idarray:
                subword=tokenizer.convert_ids_to_tokens(id)
                if subword.startswith('##'):
                    subwordconcat=subwordconcat+subword[2:]
                else:
                    subwordconcat=subword
            batch_whole_word_array.append(subwordconcat)

        tempstr = batch_whole_word_array[-1] #如果有英文，把英文小写
        if (u'\u0041' <= tempstr <= u'\u005a') or (u'\u0061' <= tempstr <= u'\u007a'):
            batch_whole_word_array[-1] = tempstr.lower()
        batch_whole_word_array.append(batch_whole_word_array)

    return batch_whole_word_array