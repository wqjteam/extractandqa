# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import json
import random

import pandas
import pandas as pd
import numpy as np


qa_json = pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)
ner_json_withkeyword = qa_json.loc[:, ["sentence",'keyword']]
ner_json_withkeyword.to_csv('data/origin/intercontest/passage_ner_withkeyword.txt',index=False,header=0)


ner_json_withoutkeyword = qa_json.loc[:, ["sentence"]]
ner_json_withoutkeyword.to_csv('data/origin/intercontest/passage_nerwithoutkeyword.txt',index=False,header=0)


qa_json = pd.read_json("data/origin/国家级非物质文化遗产代表性项目名录.json", orient='records', lines=True)
ner_json = qa_json.loc[:, ["content"]]
ner_json.to_csv('data/origin/intercontest/feiyi_passage_ner.txt',index=False,header=0)

qa_withoutkeyword = qa_json.loc[:, ["content"]]
qa_withoutkeyword.to_csv('data/origin/intercontest/feiyi_passage_qa_withoutkeyword.txt',index=False,header=0)

qa_withkeyword = qa_json.loc[:, ["content"],['name']]
qa_withkeyword.to_csv('data/origin/intercontest/feiyi_passage_qa_withkeyword.txt',index=False,header=0)
