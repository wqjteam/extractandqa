import pandas as pd
from torch.optim import AdamW
from transformers import AutoTokenizer

from PraticeOfTransformers.CustomModelForNSPQA import BertForUnionNspAndQA
from PraticeOfTransformers.DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForUnionNspAndQA.from_pretrained(model_name, num_labels=2)  # num_labels 测试用一下，看看参数是否传递
batch_size = 8
epoch_size = 10
# 用于梯度回归
optim = AdamW(model.parameters(), lr=5e-5)  # 需要填写模型的参数

# model = BertForUnionNspAndQA.from_pretrained(model_name)
print(model)
# data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
#                                                              mlm=True,
#                                                              mlm_probability=0.15,
#                                                              return_tensors="pt")

data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
                                                       mlm=True,
                                                       mlm_probability=0.15,
                                                       return_tensors="pt")

'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).drop("spos", axis=1)

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

passage_keyword_json = passage_keyword_json.explode("q_a").values
