# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import json
import random

import pandas
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
# qa_json.to_json('data/origin/intercontest/qa_keyword.json', force_ascii=False,orient='records', lines=True)


# 进行双层的for循环
# 将问题与文章结合
def passagejoinqa(passage_json):
    qaarray = []

    keyword = set()
    for tup in zip(qa_json['q'], qa_json['a'], qa_json['keyword']):
        if tup[2] in passage_json['sentence']:
            json_dict = {}
            json_dict['question'] = tup[0].strip()
            json_dict['answer'] = tup[1].strip()
            qaarray.append(json_dict)
            keyword.add(tup[2].strip())
    if len(qaarray) == 0 and passage_json['spos'] == passage_json['spos']:
        # 对于匹配不上,且自己有spos字段的进行处理\
        for pjss in passage_json['spos']:
            json_dict = {}
            json_dict['question'] = pjss.get('s').strip() + '的' + pjss.get("p").strip() + '?'
            json_dict['answer'] = pjss.get("o").strip()
            qaarray.append(json_dict)
            keyword.add(pjss.get("s").strip())

    result = qaarray, keyword
    return result


passage_json[['q_a', 'keyword']] = passage_json.apply(passagejoinqa, axis=1, result_type='expand')

# 写入数据库中
# with engine.begin() as conn:
#     pass
# 数据已经组织好，无需再次插入
# qa_json.to_sql(name='qa', con=conn, if_exists='replace', index=False)
# 读取

# passage_json.to_sql(name='passage_qa_keyword', con=conn, if_exists='replace', index=False)
# passage_json.to_json('data/origin/intercontest/passage_qa_keyword.json', force_ascii=False,orient='records', lines=True)
# qa_json=pd.read_json("data/origin/intercontest/qa_keyword.json", orient='records', lines=True)


# passage_qa_keyword_json 是可以用于训练的
passage_qa_keyword_json = pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)
# print(passage_qa_keyword_json.head())
passage_qa_keyword_json['nsp']=passage_qa_keyword_json.apply(lambda x: 0, axis=1)

sentence_df = pd.DataFrame(passage_qa_keyword_json['sentence'], index=passage_qa_keyword_json.index,
                           columns=['sentence'])

default_sentence_df=passage_qa_keyword_json.copy(deep=True)
#将索引添加为列，便于计算
sentence_df['new_temp_index']=passage_qa_keyword_json.index
index_size=passage_qa_keyword_json.index.size

q_a_spos_keyword_df = pd.DataFrame(passage_qa_keyword_json[['q_a', 'keyword', 'spos']],
                                   index=passage_qa_keyword_json.index,
                                   columns=['q_a', 'keyword', 'spos'])


# 故意新增脏数据
def match_error_multiple(sentence):
    #获取需要去除的index
    current_index=sentence['new_temp_index']
    #生成所有备选index，移除现在的index，然后在其中随机选择
    alternativearray=np.arange(0,index_size).tolist()
    alternativearray.remove(current_index)
    randomindex=random.randrange(len(alternativearray))
    q_a_spos_keyword=q_a_spos_keyword_df.iloc[randomindex]
    return q_a_spos_keyword['spos'],q_a_spos_keyword['q_a'],q_a_spos_keyword['keyword'],1

sentence_df[['spos','q_a','keyword','nsp']]=sentence_df.apply(match_error_multiple, axis=1, result_type='expand')

passage_qa_keyword_union_negate=pd.concat([default_sentence_df,sentence_df.drop("new_temp_index", axis=1)],ignore_index=True)
passage_qa_keyword_union_negate= passage_qa_keyword_union_negate.sample(frac=1) #乱序处理
passage_qa_keyword_union_negate.to_json('data/origin/intercontest/passage_qa_keyword_union_negate.json', force_ascii=False,orient='records', lines=True)
passage_keyword_json = pd.read_json("./data/origin/intercontest/passage_qa_keyword_union_negate.json", orient='records',
                                    lines=True).head(100).drop("spos", axis=1)
print(passage_keyword_json.head())