from gensim.models import Word2Vec
import pandas as pd

trainingFile = '../cutTrainingData.csv'
embeddingScore = '../data/embeddingScore.csv'

data = pd.read_csv(trainingFile, delimiter=',')
model = Word2Vec.load("word2vec_model")

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

data['embedding'] = score
data = data.drop('title1', axis = 1)
data = data.drop('title2', axis = 1)

data.to_csv(embeddingScore, index = None, header = True)