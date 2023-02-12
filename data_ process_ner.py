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

qa_json = pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)
ner_json = qa_json.loc[:, ["sentence"]]
ner_json.to_csv('data/origin/intercontest/passage_ner.txt',index=0,header=0)
