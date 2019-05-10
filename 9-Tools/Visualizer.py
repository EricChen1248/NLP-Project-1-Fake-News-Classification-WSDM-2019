import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
'''
embeddingScore = './data/embeddingScore.csv'
cats = ['unrelated', 'agreed', 'disagreed']
alphas = {'unrelated': 0.5, 'agreed': 0.5, 'disagreed':0.5}
colors = {'unrelated': 'red', 'agreed': 'blue', 'disagreed':'green'}

data = pd.read_csv('./data/train.csv')
embedding = np.load('./data/embedding_train.npy')
data['score'] = embedding

data = data[data.score != float('inf')]

for cat in cats:
    d = data[data.label==cat]
    plt.hist(d['score'], color = colors[cat], alpha=alphas[cat], bins = 50)

plt.show()
'''


data = pd.read_csv('./data/train.csv')
count = data['label'].value_counts()
bar = plt.bar(count.index, count)
bar[0].set_color('#ff8080')
bar[1].set_color('#8080ff')
bar[2].set_color('g')
plt.show()