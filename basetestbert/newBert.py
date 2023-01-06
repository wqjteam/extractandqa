import math
import re
import string
from random import randrange, shuffle, random, randint

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.utils import data
from transformers import BertForQuestionAnswering
from transformers import BertForNextSentencePrediction

# text = (
#     'Hello, how are you? I am Romeo \\n'
#     'Hello, Romeo My name is Juliet. Nice to meet you\\n'
#     'Nice meet you too. How are you today?\\n'
#     'Great. My baseball team won the competition\\n'
#     'Oh Congratulations, Juliet\\n'
#     'Thanks you Romeo'
# )

# text = (
#     'Hello, how are you? I am Romeo'
# )

passage_keyword_json = pd.read_json("../data/origin/intercontest/passage_qa_keyword.json", orient='records',
                                    lines=True).head(100).drop("spos", axis=1)
# keyword = ""

# sentences = re.sub("《汉书》、（）[～—.,!?]", ' ', text.lower()).split('\\n')  # filter '.', ',', '?', '!'
# 获取所有的单词
word_list = list()

passage_keyword_json = passage_keyword_json.explode("q_a")


def strToWord(strs):
    strlist = []
    length = len(strs)
    front = 0

    while(front < length):
        if strs[front] in string.ascii_lowercase + string.ascii_uppercase or strs[front].isdigit():
            numberordigitrear = front
            while(front < length and (strs[front] in string.ascii_lowercase + string.ascii_uppercase or strs[front].isdigit())):
                front+=1
            strlist.append(strs[numberordigitrear:front])
        else:
            strlist.append(strs[front])
            front+=1


    return strlist


# strToWord("刊1981年第3期")
regx = "[.,!?]"
regx = "([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u007a])"


def getAllWord(passage_keyword_json):
    sentence = re.sub(regx, '', passage_keyword_json["sentence"].lower())
    word_list.extend(strToWord(sentence))
    for keyword in passage_keyword_json["keyword"]:
        keyword = re.sub(regx, '', keyword.lower())
        word_list.extend(strToWord(keyword))

    pjss = passage_keyword_json["q_a"]
    question = re.sub(regx, '', pjss.get('question').lower())
    word_list.extend(strToWord(question))
    answer = re.sub(regx, '', pjss.get('answer').lower())
    word_list.extend(strToWord(answer))


passage_keyword_json.apply(getAllWord, axis=1)
# Token Purpose
# [CLS] The first token is always classification
# [SEP] Separates two sentences
# [END] End the sentence.
# [PAD] Use to truncate the sentence with equal length.
# [MASK] Use to create a mask by replacing the original word.


word2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
# 将数据存入word2id中
for i, w in enumerate(word_list):
    word2id[w] = i + 4
id2word = {i: w for i, w in enumerate(word2id)}
vocab_size = len(word2id)

# 将单词转为id  存为token
context_tokens = list()
quetion_tokens = list()
next_tokens = list()
keyword_tokens = list()
start_postion_tokens = list()
end_postion_tokens = list()
tempkeywordset=set()

def getAllTokens(passage_keyword_json):
    sentence = re.sub(regx, '', passage_keyword_json["sentence"].lower())
    context_tokens.append([word2id[s] for s in strToWord(sentence)])

    keywords = []
    for keyword in passage_keyword_json["keyword"]:
        keyword = re.sub(regx, '', keyword.lower())
        # 默认先取第一个关键词
        if keyword not in tempkeywordset:
            keyword_tokens.append([word2id[s] for s in strToWord(keyword)])
            tempkeywordset.add(keyword)

        # keyword_start_index = sentence.index(keywords)
        # keyword_end_index = keyword_start_index + len(keywords)
        # keywords.append([keyword_start_index, keyword_end_index])

    pjss = passage_keyword_json["q_a"]
    question = re.sub(regx, '', pjss.get('question').lower())
    quetion_tokens.append([word2id[s] for s in strToWord(question)])
    answer = re.sub(regx, '', pjss.get('answer').lower())
    # 获取index
    start_index = sentence.find(answer)
    start_postion_tokens.append(start_index)
    end_postion_tokens.append(start_index + len(answer))

    # 这里还没有随机数据，默认是next_tokens为true,1代表true，0代表false
    next_tokens.append(True)




passage_keyword_json.apply(getAllTokens, axis=1)

# BERT Parameters
maxlen = 30
batch_size = 6
keyword_length=10
max_pred = 5+ keyword_length # max tokens of prediction
n_layers = 6  # encoder层数
n_heads = 12  # 多头注意力机制
d_model = 768
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2


def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_quetion_tokens_index, tokens_context_tokens_index = randrange(len(quetion_tokens)), randrange(
            len(context_tokens))

        tokens_question, tokens_context = quetion_tokens[tokens_quetion_tokens_index], context_tokens[
            tokens_context_tokens_index]
        input_ids = [word2id['[CLS]']] + tokens_question + [word2id['[SEP]']] + tokens_context + [word2id['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_question) + 1) + [1] * (len(tokens_context) + 1)
        # MASK LM
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids) if
                          token != word2id['[CLS]'] and token != word2id['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        # 首先判断mask的位置是否存在keyword中
        ultimate_mask_pos = []
        front_index=0
        # while(front_index<n_pred):
        for pos in cand_maked_pos[:n_pred]:
            # if pos in keyword:
            #     pass
            for keyword_token in keyword_tokens:
                if input_ids[pos] in keyword_token:
                    pos_index = keyword_token.index(input_ids[pos])
                    pos_start_index = pos - pos_index
                    pos_end_index = pos + (len(keyword_token) - 1 - pos_index)
                    if pos_start_index >= 0 and pos_end_index < len(input_ids) and input_ids[
                                                                                   pos_start_index:pos_end_index+1] == keyword_token:
                        # 判断这个关键词是否已经存在与ultimate_mask_pos，存在则不加入了
                        # 因为存的话一次性全部存入，所以拿第一个字判断是否已经存入,
                        if pos_start_index not in set(ultimate_mask_pos):
                            ultimate_mask_pos.extend([inner_pos for inner_pos in range(pos_start_index, pos_end_index)])
                        #无论是否存入，只要对应上了，则无需再次执行下面的函数
                        continue
            ultimate_mask_pos.append(pos)

        for pos in ultimate_mask_pos:
            masked_pos.append(pos)  # 被改得位置
            masked_tokens.append(input_ids[pos])  # 原来的值
            if random() < 0.8:  # 80%
                input_ids[pos] = word2id['[MASK]']  # make mask
            elif random() < 0.9 and random() > 0.8:  # 10% 变成其他的
                index = randint(4, vocab_size - 1)  # random index in vocabulary
                input_ids[pos] = index  # replace
            else:  # 10%不变
                input_ids[pos] = input_ids[pos]  # invariant

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens,要保证所有训练数据的mask数量一致
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            # 0 指的是PAD 其他的几个分隔符也是可以的，这个没有意义，只是为了batch计算的时候，统一
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # 后面的判断是为了 positive < batch_size / 2样本均衡  尾部添加startposition 和end postition，这数据不够，的再加数据
        if tokens_quetion_tokens_index == tokens_context_tokens_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True,
                          start_postion_tokens[tokens_quetion_tokens_index],
                          end_postion_tokens[tokens_quetion_tokens_index]])  # IsNext
            positive += 1
        elif tokens_quetion_tokens_index != tokens_quetion_tokens_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False, 0, 0])  # NotNext
            negative += 1

    return batch


batch = make_batch()
input_ids, segment_ids, masked_tokens, masked_pos, isNext, start_positions, end_positions = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext, start_positions, end_positions = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids), \
    torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos), \
    torch.LongTensor(isNext), torch.LongTensor(start_positions), torch.LongTensor(end_positions)


class MyDataSet(data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext, start_positions, end_positions):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx], self.start_positions[idx], self.end_positions[idx]


loader = data.DataLoader(
    MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext, start_positions, end_positions), batch_size,
    True)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


# 用于q 相乘q 除以 根号dk 然后把结果乘以 v 求Z的
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual)  # output: [batch_size, seq_len, d_model]


# 这里比较简单
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)

        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs





class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(  # =池化
            # 一层全连接
            nn.Linear(d_model, d_model),
            nn.Dropout(0.1),
            nn.Tanh(),
        )

        # nextpredict部分使用
        self.classifier = nn.Linear(d_model, 2)

        # qa部分使用
        self.qa_outputs = nn.Linear(d_model, 2)

        # 给mlm使用，用来获取计算模型mask的结果
        # fc2 is shared with embedding layer
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        outputs = self.embedding(input_ids, segment_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            # 将上一层得 encoder得输出丢入下一层得encode中
            outputs = layer(outputs, enc_self_attn_mask)

        """
        处理next sentence
        """
        # it will be decided by first token(CLS)
        # outputs[:, 0] 也就是cls，字段进行池化
        h_pooled = self.fc(outputs[:, 0])  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] predict isNext

        """
        处理question_answer
        """
        logits = self.qa_outputs(outputs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        """
        处理mlm的结果
        """
        # 取出最后一层的encode的输出，开始前面做了mask，需要把mask还原，然后计算maxk的loss
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(outputs, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf, (start_logits, end_logits)


model = BERT()


optimizer = optim.Adadelta(model.parameters(), lr=0.001)

# 训练代码
for epoch in range(100):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext, start_positions, end_positions in loader:
        logits_lm, logits_clsf, logits_qa = model(input_ids, segment_ids, masked_pos)
        # 对于0位置的不进行loss计算,在if max_pred > n_pred:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 语言模型loss
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # for masked LM
        loss_lm = (loss_lm.float()).mean()

        # cls 也就是nextpredict
        loss_clsf = criterion(logits_clsf, isNext)  # for sentence classification

        # qa的loss
        (start_logits, end_logits) = logits_qa
        ignored_index = start_logits.size(1)
        # start_positions = start_positions.clamp(0, ignored_index)
        # end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss_qa = (start_loss + end_loss) / 2

        loss = loss_lm + loss_clsf + loss_qa
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext, start_positions, end_positions = batch[0]
# print(text)
print([id2word[w] for w in input_ids if id2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                               torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ', [pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ', True if logits_clsf else False)
