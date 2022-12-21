# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Mapping, List, Union, Dict

import numpy as np
import torch
import torch.utils.data as Data
import transformers
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel
from transformers.data.data_collator import _numpy_collate_batch, _torch_collate_batch, _tf_collate_batch, \
    DataCollatorMixin

# DataCollatorForLanguageModelingSpecial
from DataCollatorForLanguageModelingSpecial import DataCollatorForLanguageModelingSpecial

model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModelingSpecial(tokenizer=tokenizer,
                                                             mlm=True,
                                                             mlm_probability=0.15,
                                                             return_tensors="pt")
test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
inputing = torch.tensor(np.array([test[i:i + 3] for i in range(10)]))
target = torch.tensor(np.array([test[i:i + 1] for i in range(10)]))
print(inputing.shape)
print(target.shape)
train_dataset = Data.TensorDataset(inputing, target)

train_dataloader = Data.DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=10
)

for returndata in train_dataloader:
    print(returndata)
