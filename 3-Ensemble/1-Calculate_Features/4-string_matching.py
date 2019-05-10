import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import jieba.posseg as pseg
import os
import keras
from sklearn.tree import tree
import random
from fuzzywuzzy import fuzz

TRAIN_CSV_PATH = '../data/train.csv'
TEST_CSV_PATH = '../data/test.csv'
TOKENIZED_TRAIN_CSV_PATH = None

train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
#把空值填上
train.title2_zh.fillna('UNKNOWN', inplace=True)

#生成4個feature
#overlap_ratio = overlap ratio of string matching 
#partial_ratio = partial overlap ratio of string matching
#tokenset_ratio = token set ratio matching
#rumor = word Rumours
FEATURE_NAMES = ['overlap_ratio','partial_ratio','tokenset_ratio','rumor']

feature = pd.DataFrame(columns=FEATURE_NAMES)
for x1,x2 in zip(train['title1_zh'],train['title2_zh']):
    overlap_ratio = fuzz.ratio(x1,x2)/100
    partial_ratio = fuzz.partial_ratio(x1, x2)/100
    tokenset_ratio = fuzz.token_set_ratio(x1,x2)/100
    if '谣' in x1+x2:
        rumor = int(1)
    else:
        rumor = int(0)
    df = pd.DataFrame([[overlap_ratio,partial_ratio,tokenset_ratio,rumor]],columns=FEATURE_NAMES)
    feature = feature.append(df,ignore_index=True)
feature.to_csv('../data/string_train.csv',index=True)

#生成測試資料的feature
test = pd.read_csv(TEST_CSV_PATH, index_col='id')
#把空值填上
test.title2_zh.fillna('UNKNOWN', inplace=True)
#生成feature
test_feature = pd.DataFrame(columns=FEATURE_NAMES)
for x1,x2 in zip(test['title1_zh'],test['title2_zh']):
    overlap_ratio = fuzz.ratio(x1,x2)/100
    partial_ratio = fuzz.partial_ratio(x1, x2)/100
    tokenset_ratio = fuzz.token_set_ratio(x1,x2)/100
    if '谣' in x1+x2:
        rumor = int(1)
    else:
        rumor = int(0)
    df = pd.DataFrame([[overlap_ratio,partial_ratio,tokenset_ratio,rumor]],columns=FEATURE_NAMES)
    test_feature = test_feature.append(df,ignore_index=True)
test_feature.to_csv('../data/string_test.csv',index=True) 
