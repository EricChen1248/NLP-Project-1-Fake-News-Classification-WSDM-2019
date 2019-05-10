import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
import keras
from fuzzywuzzy import fuzz

train = pd.read_csv("./cutTrainingData.csv")
train.title2.fillna('UNKNOWN', inplace=True)
is_disagreed = (train["label"] == "disagreed")
disagreed_data = train[is_disagreed]



disagree_corpus_2 = []
for sentence in disagreed_data["title2"]:
    words = sentence.split("/")
    str = ' '
    new_sen = str.join(words)
    disagree_corpus_2.append(new_sen)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(disagree_corpus_2)
d = tokenizer.word_docs

sorted_d = sorted(d.items(), key=lambda item:item[1], reverse=True)[:30]
print("disagree title2")
print(sorted_d)

disagree_corpus_1 = []
for sentence in disagreed_data["title1"]:
    words = sentence.split("/")
    str = ' '
    new_sen = str.join(words)
    disagree_corpus_1.append(new_sen)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(disagree_corpus_1)
d = tokenizer.word_docs

sorted_d = sorted(d.items(), key=lambda item:item[1], reverse=True)[:30]
print("disagree title1")
print(sorted_d)

#######################################################
is_agreed = (train["label"] == "agreed")
agreed_data = train[is_agreed]
# print(agreed_data)


agree_corpus_2 = []
for sentence in agreed_data["title2"]:
    # print(sentence)
    words = sentence.split("/")
    str = ' '
    new_sen = str.join(words)
    agree_corpus_2.append(new_sen)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(agree_corpus_2)
d = tokenizer.word_docs

sorted_d = sorted(d.items(), key=lambda item:item[1], reverse=True)[:30]
print("agree title2")
print(sorted_d)

agree_corpus_1 = []
for sentence in agreed_data["title1"]:
    words = sentence.split("/")
    str = ' '
    new_sen = str.join(words)
    agree_corpus_1.append(new_sen)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(agree_corpus_1)
d = tokenizer.word_docs

sorted_d = sorted(d.items(), key=lambda item:item[1], reverse=True)[:30]
print("agree title1")
print(sorted_d)

