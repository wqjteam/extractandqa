from typing import List
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from transformers import PretrainedConfig, PreTrainedModel, BertModel, BertPreTrainedModel
from pytorchcrf import CRF
# 在编写自己的配置时，需要记住以下三点:
# https://blog.csdn.net/wwlsm_zql/article/details/123822539
# 你必须继承 PretrainedConfig,
# 你的 PretrainedConfig 的_init 必须接受任何 kwargs,
# 这些 kwargs 需要传递给超类 init。
# 继承是为了确保你从 Transformers 库中获得所有的功能，而另外两个限制来自于 PretrainedConfig 的字段比你设置的字段多。当用 from_pretrained 方法重新加载配置时，这些字段需要被配置接受，然后发送到超类。
#
# 为您的配置定义一个 model_type (这里是 model_type = “ resnet”)并不是强制性的，除非您想用 auto classes 注册您的模型(参见上一节)。
#
# 完成这些之后，您就可以轻松地创建和保存您的配置，就像使用库中的任何其他模型配置一样。下面是我们如何创建一个 resnet50d 配置文件并保存它:
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.utils import ModelOutput



class NspAndQAModelOutput(ModelOutput):
    mlm_prediction_scores: torch.FloatTensor = None
    nsp_relationship_scores: torch.FloatTensor = None
    qa_start_logits: torch.FloatTensor = None
    qa_end_logits: torch.FloatTensor = None
    sequence_output: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 如果继承PreTrainedModel 需要实现 _init_weights(self, module) 方法
# 如果继承BertPreTrainedModel 则不需要实现该方法
class BertForNerAppendBiLstmAndCrf(BertPreTrainedModel):


    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super(BertForNerAppendBiLstmAndCrf, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(
            input_size=config.pooler_fc_size,  # 和bert的输出曾 也是就pool保存一直
            hidden_size=config.hidden_size // 2,  # 1024
            batch_first=True,
            num_layers=2,
            dropout=0.5,  # 0.5
            bidirectional=True #双向
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = nn.pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs