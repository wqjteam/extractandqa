# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import json
import random

import pandas
import pandas as pd
import numpy as np


qa_json = pd.read_json("data/origin/intercontest/passage_qa_keyword.json", orient='records', lines=True)
ner_json = qa_json.loc[:, ["sentence"]]
ner_json.to_csv('data/origin/intercontest/passage_ner.txt',index=False,header=0)
