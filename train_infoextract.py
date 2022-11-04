from transformers import BertTokenizerFast, BertForQuestionAnswering





tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm-ext')
model = BertForQuestionAnswering.from_pretrained('hfl/chinese-bert-wwm-ext')

# train_encodings = tokenizer(list(train_df['text'])[:], list(train_df['question_text'])[:],
#                             return_tensors='pt', truncation=True, padding=True,
#                            max_length=512)