import torch
import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.utils import data
from transformers import BertTokenizer, AutoTokenizer

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
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

list_tokens = ['']# spilltoken=tokenizer.tokenize(sentence)
# sentence="13340平方"
encoded_dict = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=list_tokens,
    # 输入文本,采用list[tuple(question,text)]的方式进行输入 zip 把两个list压成tuple的list对
    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
    max_length=128,  # 填充 & 截断长度
    truncation=True,
    padding='longest',
    return_attention_mask=True # 返回 attn. masks.
)
encoded_dict.word_ids(batch_index=0)
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
# aa=lmtokener(zip(input,temp))


#
# print(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids']))
# print(aa)
# print(tokenizer.convert_ids_to_tokens(aa['input_ids'][0]))
# print(len(sentence))
# print(len(encoded_dict['input_ids']))
# pred=[['北','京','航','空','航','天','大','学'],[],['北','京']]
# target=[['北','京','航','空','航','天','大'],[],[]]
# target=['北','京','航','空','航','天','大','学']
# target=['北','京','航','空','航','天','大','学']
# metric=Utils.get_eval(pred_arr=pred,target_arr=target)
# print(metric)


from pytorchcrf import CRF




num_tags = 5  # 实体命名识别 每个汉字可以预测多少中类型
# model = CRF(num_tags,batch_first=True)
model = CRF(num_tags,batch_first=True)
seq_length = 3  # 句子长度（一个句子有三个单词）
batch_size = 2  # batch大小 一共输入几个句子 在这里是一个 句子

hidden= torch.randn(batch_size,seq_length,num_tags) # 输入的是 batch：几个句子 ，seq_length：每个句子的长度
# hidden= torch.tensor([0,2,2],[2,1,3])
# print(hidden.shape)# torch.Size([1, 3, 5])
# 表示：一个句子 句子长度是3 每个单词的维度是 5 ，为什么是5呢？因为是为每个单词打标签，一共有五个标签 所以
# print(hidden)

mask = torch.tensor([[1,0,0],[1,-100,-100]]) # mask的意思是 有的汉字的向量 不进行标签的预测
mask=mask.gt(-3)
# mask的形状是:[batch，seq_length]
# 这句话由于torchcrf版本不同 进而 函数设置不同 batch_first=True 假设没有这句话  那么输入模型的第一个句子序列的 mask都是true，假设有这句话 就没事 ，mask是正常的
# mask的作用是：因为是中文的句子 那么每句话都要padding 一定的长度 所以 告诉模型那些是padding的
# precise=torch.tensor([[0,2,3,2,2,-100,-100,-100,-100]])
# target=torch.tensor([[0,2,3,2,0,-100,2,-100,-100]])
# target[target==-100]=4
# precise[precise==-100]=4
# a=torchmetrics.Precision(average="macro", num_classes=5,mdmc_average ='samplewise',ignore_index=4)
# print(a(precise,target=target))

datax=[0,2,3,2,2,-100,-100,-100,-100]
train_data, dev_data = data.random_split(datax, [int(len(datax) * 0.5),
                                                      len(datax) - int(
                                                          len(datax) * 0.5)],generator=torch.Generator().manual_seed(0))
print(train_data.indices)
print(dev_data.indices)
