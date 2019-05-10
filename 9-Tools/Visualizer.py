import matplotlib.pyplot as plt
import pandas as pd
import os

embeddingScore = './data/embeddingScore.csv'
cats = ['unrelated', 'agreed', 'disagreed']
alphas = {'unrelated': 0.5, 'agreed': 0.5, 'disagreed':0.5}
colors = {'unrelated': 'red', 'agreed': 'blue', 'disagreed':'green'}

# Run Calculate if no embedding score found
if not os.path.isfile(embeddingScore):
    import Calculate.CalculateEmbeddingScore
data = pd.read_csv(embeddingScore, delimiter=',')

#data = data[data.embedding != float('inf')]

'''
for cat in cats:
    d = data[data.label==cat]
    plt.hist(d['score'], color = colors[cat], alpha=alphas[cat], bins = 50)

plt.show()
'''

for cat in cats:
    d = data[data.label==cat]

plt.show()