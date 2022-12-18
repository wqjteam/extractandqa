# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Mapping, List, Union, Dict

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import _numpy_collate_batch, _torch_collate_batch, _tf_collate_batch, \
    DataCollatorMixin




