import re
from random import randrange

text = (
       'Hello, how are you? I am Romeo.n'
       'Hello, Romeo My name is Juliet. Nice to meet you.n'
       'Nice meet you too. How are you today?n'
       'Great. My baseball team won the competition.n'
       'Oh Congratulations, Julietn'
       'Thanks you Romeo'
   )

sentences = re.sub("[.,!?-]", '', text.lower()).split('n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))


#Token Purpose
# [CLS] The first token is always classification
# [SEP] Separates two sentences
# [END] End the sentence.
# [PAD] Use to truncate the sentence with equal length.
# [MASK] Use to create a mask by replacing the original word.


word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
   word_dict[w] = i + 4
   number_dict = {i: w for i, w in enumerate(word_dict)}
   vocab_size = len(word_dict)


batch_size=10

def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))

        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]]  # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    return batch

