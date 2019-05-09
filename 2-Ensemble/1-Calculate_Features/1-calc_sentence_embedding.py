from gensim.models import Word2Vec
import numpy as np
import pandas as pd

files = ['./data/cut_training.csv', './data/cut_testing.csv']
embedding = ['./data/embedding_train.npy', './data/embedding_test.npy']

model = Word2Vec.load("./data/word2vec_model")

for f, e in zip(files, embedding):
    data = pd.read_csv(f, delimiter=',')

    c = 0
    length = data.shape[0]
    score = []
    for index, row in data.iterrows():
        try:
            score.append(model.wv.wmdistance(row['title1'].split('/'), row['title2'].split('/')))
        except Exception:
            score.append(0)
        
        c += 1
        pCount = c * 100 / length
        print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

    print(' ' * 140)
    print(f"Saving to {e}")
    np.save(e, score)