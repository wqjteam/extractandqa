# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import json
import random

import pandas
import pandas as pd
import numpy as np


qa_json = pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)
a=qa_json[qa_json.apply(lambda x: len(x['keyword'])==0, axis=1)]
# def applykeyword():


a=qa_json[qa_json.apply(lambda x: len(x['keyword'])==0, axis=1)]

pd.set_option('display.max_rows', None)
print(a)
# ner_json_withkeyword = qa_json.loc[:, ["sentence",'keyword']]
# ner_json_withkeyword.to_csv('data/origin/intercontest/passage_ner_withkeyword.txt',index=False,header=0)
#
#
# ner_json_withoutkeyword = qa_json.loc[:, ["sentence"]]
# ner_json_withoutkeyword.to_csv('data/origin/intercontest/passage_nerwithoutkeyword.txt',index=False,header=0)


# fyi_json = pd.read_json("data/origin/国家级非物质文化遗产代表性项目名录.json", orient='records', lines=True)
# fyi_json = fyi_json.loc[:, ['name','gb_time','province',"content"]]
#
# def passageunion(inputdata):
#     return inputdata['name']+':'+ inputdata['gb_time']+':'+ inputdata['content']
#
# fyi_json=fyi_json.apply(passageunion, axis=1)
#
#
#
# fyi_json.to_csv('data/origin/intercontest/feiyi_passage_ner_withkeyword.txt',index=False,header=0)
#
#
# fyi_json.to_csv('data/origin/intercontest/feiyi_passage_qa_withkeyword.txt',index=False,header=0)

# qa_withkeyword = fyi_json.loc[:, ["content"],['name']]
# qa_withkeyword.to_csv('data/origin/intercontest/feiyi_passage_withkeyword.txt',index=False,header=0)
