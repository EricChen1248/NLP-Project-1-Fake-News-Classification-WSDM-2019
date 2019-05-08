import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
import keras
from fuzzywuzzy import fuzz

train = pd.read_csv("./cutTrainingData.csv")
# print(train)
is_disagreed = (train["label"] == "disagreed")
disagreed_data = train[is_disagreed]
# print(disagreed_data)
disagree_corpus_2 = []
# disagree_corpus = np.array(disagree_corpus)
for sentence in disagreed_data["title2"]:
    words = sentence.split("/")
    str = ' '
    new_sen = str.join(words)
    # print(new_sen)
    disagree_corpus_2.append(new_sen)
# print(disagree_corpus)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(disagree_corpus_2)
d = tokenizer.word_docs

sorted_d = sorted(d.items(), key=lambda item:item[1], reverse=True)[:30]
print(sorted_d)

disagree_corpus_1 = []
for sentence in disagreed_data["title1"]:
    words = sentence.split("/")
    str = ' '
    new_sen = str.join(words)
    # print(new_sen)
    disagree_corpus_1.append(new_sen)
# print(disagree_corpus)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(disagree_corpus_1)
d = tokenizer.word_docs

sorted_d = sorted(d.items(), key=lambda item:item[1], reverse=True)[:30]
print(sorted_d)

