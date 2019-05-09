from gensim.models import Word2Vec
import json

dataFile = './data/cut_train_en.json'

with open(dataFile, 'r') as f:
    data = json.load(f)

model = Word2Vec(data.values(), size=160, window=2, min_count=5, workers=8)
model.save("./data/word2vec_model_en")