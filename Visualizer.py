import matplotlib.pyplot as plt
import pandas as pd
import os

embeddingScore = './data/embeddingScore.csv'
cats = ['unrelated', 'agreed', 'disagreed']
alphas = {'unrelated': 0.05, 'agreed': 0.05, 'disagreed':0.25}
colors = {'unrelated': 'red', 'agreed': 'blue', 'disagreed':'green'}

# Run Calculate if no embedding score found
if not os.path.isfile(embeddingScore):
    import CalculateEmbeddingScore
data = pd.read_csv(embeddingScore, delimiter=',')

data = data[data.embedding != float('inf')]

'''
for cat in cats:
    d = data[data.label==cat]
    plt.hist(d['score'], color = colors[cat], alpha=alphas[cat], bins = 50)

plt.show()
'''

sent = pd.read_csv('./data/sentimentScore.csv', delimiter=',')
sent = sent.drop('label', axis=1)

data = pd.merge(data, sent, on='id')

for cat in cats:
    d = data[data.label==cat]
    plt.scatter(x=d['embedding'], y=d['sentiment'], c=colors[cat], alpha=alphas[cat])
    #plt.show()
    #plt.hist(x=d['sent'], color=colors[cat], alpha=alphas[cat])
    #plt.show()

plt.show()