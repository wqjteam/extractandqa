# encoding=utf-8
# 由于数据格式不同，该类用于处理数据
import json
import random

import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.dialects.mysql import pymysql

cmrcdata = pd.read_json('./data/origin/cmrc/cmrc2018_trial.json')

cmrcdata['data'].apply(lambda x: x.get('paragraphs'))
