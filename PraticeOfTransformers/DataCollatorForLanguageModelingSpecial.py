# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Mapping, List, Union, Dict

import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import   DataCollatorMixin


@dataclass
class DataCollatorForLanguageModelingSpecial(DataCollatorMixin):
    """
      Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
      are not all of the same length.

      Args:
          tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
              The tokenizer used for encoding the data.
          mlm (`bool`, *optional*, defaults to `True`):
              Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
              with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
              tokens and the value to predict for the masked token.
          mlm_probability (`float`, *optional*, defaults to 0.15):
              The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
          pad_to_multiple_of (`int`, *optional*):
              If set will pad the sequence to a multiple of the provided value.
          return_tensors (`str`):
              The type of Tensor to return. Allowable values are "np", "pt" and "tf".

      <Tip>

      For best performance, this data collator should be used with a dataset having items that are dictionaries or
      BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
      [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

      </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)



    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        examples,kyewords=list(zip(*examples))  #更改过源码，进行mask的时候已经变成list（tuple（））
        examples=list(examples) #更改过源码 转为list
        kyewords=list(kyewords) #更改过源码 转为list
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            #直接执行到这
            batch = {
                "input_ids": self._torch_collate_batch(examples=examples, tokenizer= self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask,special_keyword_mask=kyewords
            ) # mask language model 为True的时候 运行这里
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None,special_keyword_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch


        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability) # special_tokens_mask相关的直接填充为0 表示在进行伯努利抽样的时候概率均为mlm_probability
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # special_tokens_mask相关的直接填充为0 表示在进行伯努利抽样的时候没有几率转为1
        masked_indices = torch.bernoulli(probability_matrix).bool()  # 每个元素都有是单一抽样的，都有probability_matrix的几率为1  .bool()将1 转为true 0 转为false
        prepare_mask_index=np.argwhere(masked_indices.numpy()==True) #torch的版本太低 没找到argwhere函数


        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK]) 使用
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _torch_collate_batch(self,examples: List[Union[List[int], Any, Dict[str, Any]]], tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0]:] = example
        return result

