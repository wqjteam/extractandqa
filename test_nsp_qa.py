from functools import partial

import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer
import torch.utils.data as Data
from PraticeOfTransformers.CustomModel import BertForUnionNspAndQA

#这个token没变化过
model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BertForUnionNspAndQA()
model.load_state_dict(torch.load("model/path1"))
model.eval() #==  self.train(False) ，使用评估模式（model.train()训练模式） 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。


'''
*******************  加载预处理数据  在test 中 预处理部分和训练阶段得代码相同  *******************
'''
batch_size = 2

# 用于梯度回归
optim = AdamW(model.parameters(), lr=5e-5)


print(model)


'''
获取数据
'''
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate_test.json", orient='records',
                                    lines=True).head(100).drop("spos", axis=1)

passage_keyword_json = passage_keyword_json[passage_keyword_json['q_a'].apply(lambda x: len(x) >= 1)]

passage_keyword_json = passage_keyword_json.explode("q_a").values



nsp_id_label = {1: True, 0: False}

nsp_label_id = {True: 1, False: 0}


def create_batch(data, tokenizer):
    text, question_answer, keyword, nsp = zip(*data)  # arrat的四列 转为tuple
    text = list(text)  # tuple 转为 list0
    questions = [q_a.get('question') for q_a in question_answer]
    answers = [q_a.get('answer') for q_a in question_answer]
    nsps = list(nsp)  # tuple 转为list

    keywords = [kw[0] for kw in keyword] # tuple 转为list 变成了双重的list 还是遍历转
    nsp_labels = []  # 用作判断两句是否相关
    start_positions_labels = []  # 记录起始位置
    end_positions_labels = []  # 记录终止始位置
    for array_index, textstr in enumerate(text):
        start_in = textstr.find(answers[array_index])
        if start_in != -1 and nsps[array_index] == 1:  # 判断是否存在
            nsp_labels.append(nsp_label_id.get(True))
            start_positions_labels.append(start_in + 1)  # 因为在tokenizer.batch_encode_plus中转换的时候添加了cls
            end_positions_labels.append(start_in + 1 + len(answers[array_index]))
        else:
            nsp_labels.append(nsp_label_id.get(False))
            #若找不到，则将start得位置 放在最末尾的位置 pad或者 [SEP]
            start_positions_labels.append(len(textstr)) #应该是len(textstr) -1得 但是因为在tokenizer.batch_encode_plus中转换的时候添加了cls 所以就是len(textstr) -1 +1
            end_positions_labels.append(len(textstr))

    # start_positions = [q_a.get('start_position') for q_a in question_answer]
    # end_positions = [q_a.get('end_position') for q_a in question_answer]
    '''
    bert是双向的encode 所以qestion和text放在前面和后面区别不大
    '''
    encoded_dict_textandquestion = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=list(zip(text, questions)),  # 输入文本对 # 输入文本,采用list[tuple(text,question)]的方式进行输入
        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        max_length=256,  # 填充 & 截断长度
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,  # 返回 attn. masks.
    )
    encoded_dict_keywords = tokenizer.batch_encode_plus(batch_text_or_text_pairs=keywords, add_special_tokens=False,
                                                    # 添加 '[CLS]' 和 '[SEP]'
                                                    pad_to_max_length=False,
                                                    return_attention_mask=False
                                                    )
    base_input_ids = [torch.tensor(input_id) for input_id in encoded_dict_textandquestion['input_ids']]
    attention_masks = [torch.tensor(attention) for attention in encoded_dict_textandquestion['attention_mask']]
    # 传入的参数是tensor形式的input_ids，返回input_ids和label，label中-100的位置的词没有被mask


     # 需要获取不是-100的位置，证明其未被替换，这也是target -100的位置在计算crossentropyloss 会丢弃

    # 对于model只接受tensor[list] 必须为 list[tensor] 转为tensor[list]
    return base_input_ids, torch.stack(
        attention_masks), torch.tensor(nsp_labels), torch.tensor(
        start_positions_labels), torch.tensor(end_positions_labels)


# 把一些参数固定
create_batch_partial = partial(create_batch, tokenizer=tokenizer)

# batch_size 除以2是为了 在后面认为添加了负样本   负样本和正样本是1：1
train_dataloader = Data.DataLoader(
    passage_keyword_json, shuffle=True, collate_fn=create_batch_partial, batch_size=batch_size
)


# 进行训练
for return_batch_data in train_dataloader:
    base_input_ids, attention_masks, nsp_labels, start_positions_labels, end_positions_labels = return_batch_data
    model_output = model(input_ids=base_input_ids, attention_mask=attention_masks)
    config = model.config
    prediction_scores = model_output.mlm_prediction_scores
    nsp_relationship_scores = model_output.nsp_relationship_scores
    qa_start_logits = model_output.qa_start_logits
    qa_end_logits = model_output.qa_end_logits

    '''
    loss的计算
    '''
    loss_fct = CrossEntropyLoss()  # -100 index = padding token 默认就是-100

    '''
    nsp loss 计算
    '''
    nsp_loss = loss_fct(nsp_relationship_scores.view(-1, 2), nsp_labels.view(-1))

    '''
    qa loss 计算
    '''
    start_loss = loss_fct(qa_start_logits, start_positions_labels)
    end_loss = loss_fct(qa_end_logits, end_positions_labels)
    qa_loss = (start_loss + end_loss) / 2

    total_loss = nsp_loss+ qa_loss


