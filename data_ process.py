# 由于数据格式不同，该类用于处理数据
import json

import pandas

qa_json = pandas.read_json("data/origin/intercontest/question_train.json", orient='records', lines=True)
print(qa_json.loc[:, ["q", "a"]].head(10))

passage_json = pandas.read_json("data/origin/intercontest/all_passages.json", orient='records', lines=True)
print(passage_json.loc[:,["sentence","spos"]].head(10))
