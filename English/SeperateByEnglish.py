import nltk
import pandas
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import json

data = pandas.read_csv('./data/train.csv')
nouns = set(['NN', 'NNS', 'NNP', 'NNPS'])

with open('cutEnglish.json', 'r') as f:
    sent = json.load(f)

model = Word2Vec.load('word2vec_model_en')

scores = []
c = 0
for _, row in data.iterrows():
    id1 = row['tid1']
    id2 = row['tid2']
    score = 0
    tags1 = nltk.pos_tag(sent[str(id1)])
    tags2 = nltk.pos_tag(sent[str(id2)])
    count = 0
    for w1 in tags1:
        if w1[1] in nouns and w1[0] in model.wv:
            for w2 in tags2:
                if w2[1] in nouns and w2[0] in model.wv:
                    score += model.wv.similarity(w1[0], w2[0])
                    count += 1

    scores.append(score / count if count != 0 else 0)
    
    c += 1
    pCount = c * 100 / data.shape[0]
    print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

df = pandas.DataFrame(scores, columns=['EnglishNounScore'])
df.index.name = 'index'
df.to_csv('englishNoun.csv')
