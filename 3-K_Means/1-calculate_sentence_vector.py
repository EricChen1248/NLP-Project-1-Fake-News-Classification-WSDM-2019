import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


#%%
model = Word2Vec.load("./data/word2vec_model")

dataFile = "./data/cut_training.csv"
data = pd.read_csv(dataFile, delimiter=',')

#%%
ids = [['id1', 'title1'], ['id2', 'title2']]
vectors = {}
c = 0
for _, row in data.iterrows():
    for id, title in ids:
        if row[id] not in vectors:
            try:
                vectors[row[id]] = np.mean([model.wv[w] if w in model.wv else 0 for w in row[title].split('/')], axis=0).tolist()
            except:
                pass
        
    c += 1
    pCount = c * 100 / data.shape[0]
    print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

print(' ' * 140)
#%%
np.save('./data/sentence_vectors.npy', vectors)
    