#%%
import pandas as pd
import numpy as np
import json

#%%
dataFile = "./data/cut_training.csv"
data = pd.read_csv(dataFile, delimiter=',')

vectors = np.load('./data/sentence_vectors.npy').item() 
#%%
class group:
    def __init__(self, vect, id):
        self.vector = np.array(vect)
        self.ids = [id]

    def combine(self, other):
        self.vector = self.vector + other.vector
        self.ids += other.ids
        other.ids = None
        other.vector = None

def similarity(g1, g2):
    return np.dot(g1.vector, g2.vector) /np.linalg.norm(g1.vector) / np.linalg.norm(g2.vector)

#%%

vectors = {int(k):group(v, int(k)) for k,v in vectors.items()}

#%%
changed = True
i = 0
rows = range(len(data))
while changed:
    i += 1
    changed = False
    c = 0
    
    newRow = []
    length = len(rows)
    print(f"Iteration {i} with {length} rows")
    for r in rows:
        row = data.iloc[r]
        try:
            if similarity(vectors[row['id1']], vectors[row['id2']]) > 0.9 :
                vectors[row['id1']].combine(vectors[row['id2']])
                vectors[row['id2']] = vectors[row['id1']]
                changed = True
            else:
                newRow.append(r)
        except:
            pass
        
        c += 1
        pCount = c * 100 / len(rows)
        print(f"Finding Neighbours: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 
    
    rows = newRow
    print(' ' * 140, end='\r')

score = np.zeros(data.shape[0])
c = 0
for _, row in data.iterrows():
    try:
        score[c] = similarity(vectors[row['id1']],vectors[row['id2']])
    except:
        pass

    
    c += 1
    pCount = c * 100 / data.shape[0]
    print(f"Calculating Scores: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

print(' ' * 140)
np.save('./data/k-means-score.npy', score)