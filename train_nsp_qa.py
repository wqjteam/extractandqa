# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Mapping, List, Union, Dict

import numpy as np
import torch
import torch.utils.data as Data
import transformers
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel
from transformers.data.data_collator import _numpy_collate_batch, _torch_collate_batch, _tf_collate_batch, \
    DataCollatorMixin, DataCollatorForLanguageModeling

# DataCollatorForLanguageModelingSpecial
from DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
#                                                              mlm=True,
#                                                              mlm_probability=0.15,
#                                                              return_tensors="pt")

data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
                                                mlm=True,
                                                mlm_probability=0.15,
                                                return_tensors="pt")
sent = "我爱北京天安门，天安门上太阳升"
question = "我爱什么"
# 创建一个实例，参数是tokenizer


encoded_dict = tokenizer.encode_plus(
    sent,  # 输入文本
    question,  #
    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
    max_length=32,  # 填充 & 截断长度
    truncation=True,
    pad_to_max_length=True,
    return_attention_mask=True,  # 返回 attn. masks.
)

print(encoded_dict)
input_ids = [torch.tensor(encoded_dict['input_ids'])]
# 传入的参数是tensor形式的input_ids，返回input_ids和label，label中
# -100的位置的词没有被mask，
output = data_collator(input_ids)
print(output)
# train_dataset = Data.TensorDataset(inputing, target)
# train_dataset = (inputing, target)
# train_dataloader = Data.DataLoader(
#     input_ids, shuffle=True, collate_fn=data_collator, batch_size=10
# )


# for returndata in train_dataloader:
#     print(returndata)
