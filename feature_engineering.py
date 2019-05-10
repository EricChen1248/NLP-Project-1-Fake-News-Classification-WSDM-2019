import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import jieba.posseg as pseg
import os
import keras
from sklearn.tree import tree
import random
from fuzzywuzzy import fuzz

TRAIN_CSV_PATH = './train.csv'
TEST_CSV_PATH = './test.csv'
TOKENIZED_TRAIN_CSV_PATH = None

# train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
# # 謠言詞 = ['谣', '官方', '假', '真相']
# train.title2_zh.fillna('UNKNOWN', inplace=True)

FEATURE_NAMES = ['overlap_ratio','partial_ratio','tokenset_ratio','rumor']

# feature = pd.DataFrame(columns=FEATURE_NAMES)
rumor_tokens = ['谣', '官方', '假', '真相']
# for x1,x2 in zip(train['title1_zh'],train['title2_zh']):
#     overlap_ratio = fuzz.ratio(x1,x2)/100
#     partial_ratio = fuzz.partial_ratio(x1, x2)/100
#     tokenset_ratio = fuzz.token_set_ratio(x1,x2)/100
#     rumor = int(0)
#     for token in rumor_tokens:
#         if token in x2:
#             rumor = int(1)
#             break
#     print(rumor)
#     df = pd.DataFrame([[overlap_ratio,partial_ratio,tokenset_ratio,rumor]],columns=FEATURE_NAMES)
#     feature = feature.append(df,ignore_index=True)
# feature.to_csv('feature_train.csv',index=False)

test = pd.read_csv(TEST_CSV_PATH, index_col='id')
#把空值填上
test.title2_zh.fillna('UNKNOWN', inplace=True)
feature2 = pd.DataFrame(columns=FEATURE_NAMES)
c = 0
for x1,x2 in zip(test['title1_zh'],test['title2_zh']):
    overlap_ratio = fuzz.ratio(x1,x2)/100
    partial_ratio = fuzz.partial_ratio(x1, x2)/100
    tokenset_ratio = fuzz.token_set_ratio(x1,x2)/100
    rumor = int(0)
    sX2 = set(x2)
    for token in rumor_tokens:
        if token in sX2:
            rumor = int(1)
            break
            
    c += 1
    pCount = c * 100 / test.shape[0]
    print("Calculating Score: (%s%s)%s%s" % ('#' * int(pCount), '-' * (100 - int(pCount)), round(pCount, 2), '%'), end='\r') 
    df = pd.DataFrame([[overlap_ratio,partial_ratio,tokenset_ratio,rumor]],columns=FEATURE_NAMES)
    feature2 = feature2.append(df,ignore_index=True)
feature2.to_csv('feature_test.csv',index=False)