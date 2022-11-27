# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils import data

from transformers import BertForQuestionAnswering, BertTokenizerFast

from transformers import BertForNextSentencePrediction
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW


tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm-ext')
model = BertForQuestionAnswering.from_pretrained('hfl/chinese-bert-wwm-ext')

text=["Lyon Jone is a killer of cruelty","KK is Basketball"]
question_text=["Who is Lyon Jone","What is KK"]
index_start=[7,3]
index_end=[11,4]

train_df = pd.DataFrame(list(zip(text,question_text,index_start,index_end)), columns = ['text','question_text','index_start','index_end'])

train_encodings = tokenizer(list(train_df['question_text'])[:],list(train_df['text'])[:] ,
                            return_tensors='pt', truncation=True, padding=True,
                           max_length=512)

train_encodings['start_positions'] = [train_encodings.char_to_token(idx, x) for idx, x in enumerate(train_df['index_start'].values[:])]
train_encodings['end_positions'] = [train_encodings.char_to_token(idx, x-1) for idx, x in enumerate(train_df['index_end'].values[:])]






class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = SquadDataset(train_encodings)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for idx, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        acc1 = ( (start_pred == start_positions).sum() / len(start_pred) ).item()
        acc2 = ( (end_pred == end_positions).sum() / len(start_pred) ).item()

        if idx % 10 == 0:
            print(loss.item(), acc1, acc2)
            # with codecs.open('log.log', 'a') as up:
            #     up.write('{3}\t{0}\t{1}\t{2}\n'.format(loss.item(), acc1, acc2,
            #                                            str(epoch) + '/' + str(idx) +'/'+ str(len(train_loader))))

model.eval()