import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


#%%
model = Word2Vec.load("word2vec_model")

dataFile = "./cutTrainingData.csv"
data = pd.read_csv(dataFile, delimiter=',')

#%%
ids = [['id1', 'title1'], ['id2', 'title2']]
vectors = {}
for _, row in data.iterrows():
    for id, title in ids:
        if row[id] not in vectors:
            try:
                vectors[row[id]] = np.mean([model.wv[w] if w in model.wv else 0 for w in row[title].split('/')], axis=0).tolist()
            except:
                pass

#%%
with open('vectors.json', 'w') as f:
    json.dump(vectors, f)
    