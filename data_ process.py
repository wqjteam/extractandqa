# 由于数据格式不同，该类用于处理数据
import json

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
# from pandas import Dataframe as df
import jieba

qa_json = pd.read_json("data/origin/intercontest/question_train.json", orient='records', lines=True)
qa_json = qa_json.loc[:, ["q", "a"]]

passage_json = pd.read_json("data/origin/intercontest/all_passages.json", orient='records', lines=True)
passage_json = passage_json.loc[:, ["sentence", "spos"]]


# 通过经验判断是否 将大部分的qa的关键词筛选出来，小部分的采用人工筛选的方式
def getkeyowrd(qa_json):
    question = qa_json['q']
    if '遗址' in question or '盒' in question or '服' in question or '馆' in question:
        if '遗址' in question:
            return question[:question.index('遗址') + len('遗址')]
        elif '盒' in question:
            return question[:question.index('盒') + len('盒')]
        elif '服' in question:
            return question[:question.index('服') + len('服')]
        elif '馆' in question:
            return question[:question.index('馆') + len('馆')]
    else:
        # return jieba.lcut(qa_json["q"])[0]
        return "--------"


qa_json.loc[:, "keyword"] = qa_json.apply(getkeyowrd, axis=1)

# conn = pymysql.Connect(
#   host = '192.168.4.110',
#   port = 3306,
#   user = 'root',
#   passwd = '123456',
#   db = 'relic_data',
#   charset = 'utf8'
#   )


# engine = create_engine("mysql+pymysql://root:123456@192.168.4.110:3306/relic_data?charset=utf8")

# with engine.begin() as conn:
#     pass
    # 数据已经组织好，无需再次插入
    # qa_json.to_sql(name='qa', con=conn, if_exists='replace', index=False)
    #
    # 读取
    # qa_json = pd.read_sql_table(table_name='qa', con=conn)
#qa_json.to_json('data/origin/intercontest/qa_keyword.json', force_ascii=False,orient='records', lines=True)



# 进行双层的for循环
# 将问题与文章结合
def passagejoinqa(passage_json):
    qaarray=[]

    keyword = set()
    for tup in zip(qa_json['q'], qa_json['a'], qa_json['keyword']):
        if tup[2] in passage_json['sentence']:
            json_dict = {}
            json_dict['question'] = tup[0].strip()
            json_dict['answer'] = tup[1].strip()
            qaarray.append(json_dict)
            keyword.add(tup[2].strip())
    if len(qaarray) == 0 and passage_json['spos'] == passage_json['spos']  :
        # 对于匹配不上,且自己有spos字段的进行处理\
        for pjss in passage_json['spos']:
            json_dict = {}
            json_dict['question'] = pjss.get('s').strip() + '的' + pjss.get("p").strip() + '?'
            json_dict['answer'] = pjss.get("o").strip()
            qaarray.append(json_dict)
            keyword.add(pjss.get("s").strip())

    result= qaarray, keyword
    return result


passage_json[['q_a','keyword']]= passage_json.apply(passagejoinqa, axis=1,result_type='expand')

# 写入数据库中
# with engine.begin() as conn:
#     pass
    # 数据已经组织好，无需再次插入
    # qa_json.to_sql(name='qa', con=conn, if_exists='replace', index=False)
    # 读取

    # passage_json.to_sql(name='passage_qa_keyword', con=conn, if_exists='replace', index=False)
passage_json.to_json('data/origin/intercontest/passage_qa_keyword.json', force_ascii=False,orient='records', lines=True)
#qa_json=pd.read_json("data/origin/intercontest/qa_keyword.json", orient='records', lines=True)


#passage_qa_keyword_json 是可以用于训练的
# passage_qa_keyword_json=pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)
