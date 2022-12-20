# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Mapping, List, Union, Dict

import numpy as np
import transformers
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel
from transformers.data.data_collator import _numpy_collate_batch, _torch_collate_batch, _tf_collate_batch, \
    DataCollatorMixin


# DataCollatorForLanguageModelingSpecial
model_name = 'hfl/chinese-xlnet-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                             mlm=True,
                                                             mlm_probability=0.15,
                                                             return_tensors="pt")



