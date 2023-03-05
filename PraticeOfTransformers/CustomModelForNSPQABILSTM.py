from typing import List
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from transformers import PretrainedConfig, PreTrainedModel, BertModel, BertPreTrainedModel
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
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLMPredictionHead, BertPooler
from transformers.utils import ModelOutput



class BiLstmNspQAModelOutput(ModelOutput):
    nsp_relationship_scores: torch.FloatTensor = None
    qa_start_logits: torch.FloatTensor = None
    qa_end_logits: torch.FloatTensor = None
    sequence_output: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 如果继承PreTrainedModel 需要实现 _init_weights(self, module) 方法
# 如果继承BertPreTrainedModel 则不需要实现该方法
class CustomModelForNSPQABILSTM(BertPreTrainedModel):


    #config_class = NspAndQAConfig   #可有可无 在from_pretrain 的时候就会默认加载，除非需要改变模型层数，基本册数之类的 之类的
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        # super(BertForUnionNspAndQA, self).__init__(config) 与 super().__init__(config) 二选一
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)



        self.bilstm=nn.LSTM(  # 解决问题：抽取用于实体分类的包含上下文的文本信息
            input_size=config.hidden_size,  # 和bert encode的输出层 也即是隐藏层
            hidden_size=config.hidden_size // 2,  # 隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数； 因为是双向lstm 所以除以2
            batch_first=True,
            num_layers=2,
            #  dropout=0.5,  # 0.5 默认值0，除最后一层，每一层的输出都进行dropout；
            bidirectional=True  # 双向
        )


        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pooler = BertPooler(config)

        self.nsp_cls = nn.Linear(config.hidden_size, 2)  #nsp就是两种 是和否

        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BiLstmNspQAModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids, #必须传
            attention_mask=attention_mask, #必须传
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0] #隐藏层 最后一层
        lstm_output, _ = self.bilstm(sequence_output)

        padded_sequence_output = self.dropout(lstm_output)


        #contiguous相当于 flush 是内存改变（前面相当于做了一些lazy的操作）
        #qa
        qa_logits = self.qa_outputs(padded_sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()


        #nsp
        pooled_output = self.pooler(padded_sequence_output)
        nsp_relationship_scores = self.nsp_cls(pooled_output)




        return BiLstmNspQAModelOutput(
            nsp_relationship_scores=nsp_relationship_scores,
            qa_start_logits=start_logits,
            qa_end_logits=end_logits,
            sequence_output=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
