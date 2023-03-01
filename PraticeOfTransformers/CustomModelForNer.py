from typing import List
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from transformers import PretrainedConfig, PreTrainedModel, BertModel, BertPreTrainedModel
from pytorchcrf import CRF
# 在编写自己的配置时，需要记住以下三点:
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

        self.bert = BertModel(config)  # 解决问题：NER标注数据少，文本信息抽取效果不佳

        for param in self.bert.parameters():  # bert 不更新参数
            param.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(  # 解决问题：抽取用于实体分类的包含上下文的文本信息
            input_size=config.hidden_size,  # 和bert encode的输出层 也即是隐藏层
            hidden_size=config.hidden_size // 2,  # 隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数； 因为是双向lstm 所以除以2
            batch_first=True,
            num_layers=2,
            #  dropout=0.5,  # 0.5 默认值0，除最后一层，每一层的输出都进行dropout；
            bidirectional=True  # 双向
        )
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels)  # 得到bert+bilstm预测标签

        # crf的第一个词必须是开启的，说明cls这个字符的去掉
        self.crf = CRF(config.num_labels, batch_first=True)  # 用来约束标签 实体内标签分类的一致性，T个N分类问题转化为NT的分类问题

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None,is_test=False):

        outputs = self.bert(input_ids=input_ids
                            # )
                            ,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]


        # dropout pred_label的一部分feature
        lstm_output, _ = self.bilstm(sequence_output)
        padded_sequence_output = self.dropout(lstm_output)
        # 得到判别值 标签的判别
        logits = self.classifier(padded_sequence_output)

        loss_mask = labels.gt(-1)  # 作为mask 相等于pading这类的不计算 ，这里遇到问题了 我cls的标签在第一个，这里不允许
        labels[labels == -100] = 0  # 这里是-100 计算会报错，设置为0，这样就在num_labels的范围内，但是loss_mask的设置在计算loss的时候是影响 loss
        '''
        crf 注意到这个返回值为对数似然，所以当你作为损失函数时，需要在这个值前添加负号.。默认地，这个对数似然是批上的求和。
        '''
        loss = self.crf(emissions=logits, tags=labels, mask=loss_mask, reduction='mean') * (-1)
        loss = torch.mean(loss, dim=-1)
        outputs = (loss,)
        if not is_test:
            outputs = outputs + (logits,)
        else:
            predict = self.crf.decode(logits)
            outputs = outputs + (predict,)
        return outputs

    def decode(self, input_ids, attention_mask, segment_ids):
        embeds = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)
        results = self.crf.decode(lstm_out)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result))
        return torch.stack(result_tensor)
