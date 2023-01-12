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
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.utils import ModelOutput


class NspAndQAConfig(PretrainedConfig):

    def __init__(
            self,
            attention_probs_dropout_prob: float = 0.1,
            directionality: str = "bidi",
            hidden_act: str = "gelu",
            hidden_dropout_prob: int = 0.1,
            hidden_size: int = 768,
            initializer_range: float = 0.02,
            intermediate_size: int = 3072,
            layer_norm_eps: float = 1e-12,
            max_position_embeddings: int = 512,
            model_type: str = "bert",
            num_attention_heads: int = 12,
            num_hidden_layers: int = 12,
            pad_token_id: int = 0,
            pooler_fc_size: int = 768,
            pooler_num_attention_heads: int = 12,
            pooler_num_fc_layers: int = 3,
            pooler_size_per_head: int = 128,
            pooler_type: str = "first_token_transform",
            type_vocab_size: int = 2,
            vocab_size: int = 21128,
            **kwargs,
    ):
        # if block_type not in ["basic", "bottleneck"]:
        #     raise ValueError(f"`block` must be 'basic' or bottleneck', got {block_type}..")
        # if stem_type not in ["", "deep", "deep-tiered"]:
        #     raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {block_type}..")

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.directionality = directionality
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)


NspAndQAConfig_config = NspAndQAConfig(model_type="bert")
NspAndQAConfig_config.save_pretrained("custom-bert-qa")
# 您还可以使用 PretrainedConfig 类的任何其他方法，如 push_to_Hub () ，将配置直接上传到 Hub。

NspAndQAConfig_config = NspAndQAConfig.from_pretrained("custom-bert-qa")


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
class BertForUnionNspAndQA(BertPreTrainedModel):
    config_class = NspAndQAConfig

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.nsp_cls = nn.Linear(config.hidden_size, self.num_labels)

        self.mlm_cls = BertOnlyMLMHead(config)
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
    ) -> Union[Tuple[torch.Tensor], NspAndQAModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
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

        sequence_output = outputs[0]
        #mlm
        prediction_scores = self.mlm_cls(sequence_output)
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        #qa
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()


        # if start_positions is not None and end_positions is not None:
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_positions = start_positions.clamp(0, ignored_index)
        #     end_positions = end_positions.clamp(0, ignored_index)
        #
        #     loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        #     start_loss = loss_fct(start_logits, start_positions)
        #     end_loss = loss_fct(end_logits, end_positions)
        #     total_loss = (start_loss + end_loss) / 2
        #
        # if not return_dict:
        #     output = (start_logits, end_logits) + outputs[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output

        #nsp
        pooled_output = outputs[1]
        seq_relationship_scores = self.nsp_cls(pooled_output)
        # next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
        # next_sentence_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()

        # if not return_dict:
        #     output = (seq_relationship_scores,) + outputs[2:]
        #     return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # return NextSentencePredictorOutput(
        #     loss=next_sentence_loss,
        #     logits=seq_relationship_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        return NspAndQAModelOutput(
            mlm_prediction_scores=shifted_prediction_scores,
            nsp_relationship_scores=seq_relationship_scores,
            qa_start_logits=start_logits,
            qa_end_logits=end_logits,
            sequence_output=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
