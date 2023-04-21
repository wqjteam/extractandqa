# encoding=utf-8
import datetime
import uuid

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch,helpers

es = Elasticsearch([{"host": "47.120.39.188", "port": 9200, "timeout": 1500}])
# 获取遗址相关
passage_keyword_json_realrelic = pd.read_json("./data/origin/intercontest/passage_qa_keyword.json", orient='records',
                                              lines=True)

# 获取非遗址相关
passage_keyword_json_virtualrelic_file = open("./data/origin/intercontest/feiyi_passage_ner_withkeyword.txt",
                                              encoding="utf-8")
virtualrelic_setence = []
virtualrelic_keyword = []
for row in passage_keyword_json_virtualrelic_file.readlines():
    tmp_list = row.split(':')  # 按‘:'切分每行的数据
    maxlen = 510 if len(row) > 510 else len(row)
    virtualrelic_setence.append(row[0:maxlen])
    virtualrelic_keyword.append([tmp_list[0]])
virtualrelic_setence_keyword = np.concatenate(
    (np.array(virtualrelic_setence).reshape(-1, 1), np.array(virtualrelic_keyword, dtype=list).reshape(-1, 1)), axis=1)
passage_keyword_json_virtualrelic = pd.DataFrame(data=virtualrelic_setence_keyword, columns=['sentence', 'keyword'])
passage_keyword_json_virtualrelic = passage_keyword_json_virtualrelic.loc[:, ['sentence']]

union_pd = pd.concat(
    [passage_keyword_json_realrelic.loc[:, ['sentence']], passage_keyword_json_virtualrelic], axis=0)


union_pd["uuid"]=union_pd.apply(lambda x:uuid.uuid1(), axis=1, result_type='expand')
# md5 加密
def md5(string):
    import hashlib
    # 对要加密的字符串进行指定编码
    string = string.encode(encoding='UTF-8')
    # md5加密
    return hashlib.md5(string).hexdigest()


def write_to_es():

    # 写入es
    actions = []
    for index, row in union_pd.iterrows():
        # day = datetime.datetime.strftime(row[0], '%Y-%m-%d')
        action = {
            "_index": 'culture_heritage',
            "_id": row[1],
            "_source": {
                "info": row[0],

            }
        }
        actions.append(action)

    helpers.bulk(es, actions)


#write_to_es()
body = {
    "query":{
        "match_all":{}
    }
}
# es.delete(index='culture_heritage',id="")
# res = es.indices.delete('culture_heritage')  # 删除索引
# es.search(index="culture_heritage" ,body=body)
# print(es.search(index="culture_heritage" ,body=body))
print(es.count(index="culture_heritage"))
