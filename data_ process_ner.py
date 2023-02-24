# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import json
import random

import pandas
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import pymysql

qa_json = pd.read_json(path_or_buf='data/origin/intercontest/passage_qa_keyword.json', orient='records', lines=True)

a=qa_json

pd.set_option('display.max_rows', None)
# print(a)

# conn = pymysql.Connect(
#   host = '192.168.4.144',
#   port = 3306,
#   user = 'root',
#   passwd = '123456',
#   db = 'relic_data',
#   charset = 'utf8'
#   )


engine = create_engine("mysql+pymysql://root:123456@192.168.4.144:3306/relic_data?charset=utf8")

# a.to_sql(name='user_tableq', con=engine, if_exists='replace', index=False)

#读取
a_keyword_json = pd.read_sql_table(table_name='user_tableq', con=engine).to_numpy()

def getkeyword(inputdata):
    if len(inputdata.keyword)>0:
        return inputdata.keyword
    else:
        for ssnupmy in a_keyword_json:
            if ssnupmy[0]==inputdata.sentence:
                return ssnupmy[1].split('|')
a['keyword']=a.apply(getkeyword, axis=1)

a=qa_json[qa_json.apply(lambda x: len(x['keyword'])>1, axis=1)]
print(a)
#a.to_json('data/origin/intercontest/passage_qa_keyword.json', force_ascii=False,orient='records', lines=True)
# a_json.to_json('data/origin/intercontest/qa_keyword.json', force_ascii=False,orient='records', lines=True)

# with engine.begin() as conn:
#     pass
# 数据已经组织好，无需再次插入
# qa_json.to_sql(name='qa', con=conn, if_exists='replace', index=False)
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
