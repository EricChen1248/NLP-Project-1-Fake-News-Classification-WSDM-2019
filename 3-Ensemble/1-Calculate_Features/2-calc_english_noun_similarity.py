from nltk.tag.perceptron import PerceptronTagger
import pandas
from gensim.models import Word2Vec
import numpy
from nltk.tokenize import word_tokenize
import json

files = ["./data/train.csv", "./data/test.csv"]
dest = ["./data/english_noun_score.npy", "./data/english_noun_score_test.npy"]
js = ["./data/cut_train_en.json", "./data/cut_test_en.json"]

model = Word2Vec.load('./data/word2vec_model_en')
nouns = set(['NN', 'NNS', 'NNP', 'NNPS'])
tagger = PerceptronTagger()

for f, d, j in zip(files, dest, js):
    with open(j, 'r') as jj:
        sent = json.load(jj)

    data = pandas.read_csv(f)

    scores = numpy.empty(data.shape[0])
    c = 0
    
    nouns = {}
    for _, row in data.iterrows():
        id1 = row['tid1']
        id2 = row['tid2']
        score = 0

        if id1 not in nouns:
            nouns[id1] = []
            for w in tagger.tag(sent[str(id1)]):
                if w[1] in nouns and w[0] in model.wv:
                    nouns[id1].append(w[0])

        if id2 not in nouns:
            nouns[id2] = []
            for w in tagger.tag(sent[str(id2)]):
                if w[1] in nouns and w[0] in model.wv:
                    nouns[id2].append(w[0])

        c += 1
        pCount = c * 100 / data.shape[0]
        print(f"Finding Nouns: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

    c = 0
    for _, row in data.iterrows():
        id1 = row['tid1']
        id2 = row['tid2']
        score = 0
        tags1 = nouns[id1]
        tags2 = nouns[id2]
        count = 0
        for w1 in tags1:
            for w2 in tags2:
                score += model.wv.similarity(w1[0], w2[0])
                count += 1

        scores[c] = score / count if count != 0 else 0
        
        c += 1
        pCount = c * 100 / data.shape[0]
        print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

    numpy.save(d, scores)
