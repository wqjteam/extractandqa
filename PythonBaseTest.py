import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForTokenClassification, \
    DataCollatorForWholeWordMask

import CommonUtil
from PraticeOfTransformers.DataCollatorForWholeWordMaskOriginal import DataCollatorForWholeWordMaskOriginal

pred = torch.tensor([[0.0,10.0,0.0],[0.0,0.0,10.0]])

target = torch.tensor([1, 2], dtype=torch.long)

# 函数形式：
out = F.cross_entropy(pred, target)

# 类的形式：
func = CrossEntropyLoss()  # 括号不要掉，定义一个对象
out = func(pred, target)

# a=[1,1,1,1,5,10,1,2,3]
# b=[1,1]
# index=0
# find_all_index=[]
# while(index<len(a)):
#     if a[index]==b[0] and index+len(b)<=len(a) and a[index:index+len(b)]==b[:] :
#         find_all_index.append((index,index+len(b)))
#         index+=len(b)
#     else:
#         index+=1
#
# print(find_all_index)

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence="长治市博物馆，位于长治市太行西街。1990年9月动工兴建新馆，1992年10月落成，占地面积13340平方米，建筑面积8200平方米"
sentence="13340平方"
encoded_dict = tokenizer.encode_plus(
    text=sentence,
    # 输入文本,采用list[tuple(question,text)]的方式进行输入 zip 把两个list压成tuple的list对
    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
    max_length=128,  # 填充 & 截断长度
    truncation=True,
    padding='longest',
    return_attention_mask=True,  # 返回 attn. masks.
)
# lmtokener=DataCollatorForWholeWordMask(tokenizer=tokenizer,
#                                                        mlm=True,
#                                                        mlm_probability=0.15,
#                                                        return_tensors="pt")
lmtokener=DataCollatorForWholeWordMaskOriginal(tokenizer=tokenizer,
                                                       mlm=True,
                                                       mlm_probability=0.35,
                                                       return_tensors="pt")
# lmtokener=DataCollatorForTokenClassification(tokenizer)
# input=[{'input_ids':encoded_dict['input_ids'],'token_type_ids':encoded_dict['token_type_ids'],'labels':encoded_dict['attention_mask']} ]
input=[encoded_dict['input_ids']]
temp=[list()]
aa=lmtokener(zip(input,temp))



print(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids']))
print(aa)
print(tokenizer.convert_ids_to_tokens(aa['input_ids'][0]))
print(len(sentence))
print(len(encoded_dict['input_ids']))

