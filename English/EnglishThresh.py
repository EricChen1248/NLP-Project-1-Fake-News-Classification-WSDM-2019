import pandas
import matplotlib.pyplot as plt


dataFile = './data/train.csv'
nounFile = './englishNoun.csv'

data = pandas.read_csv(dataFile)
nouns = pandas.read_csv(nounFile)

data['nounScore'] = nouns['EnglishNounScore']

'''
cats = ['unrelated', 'agreed', 'disagreed']
alphas = {'unrelated': 0.5, 'agreed': 0.5, 'disagreed':0.5}
colors = {'unrelated': 'red', 'agreed': 'blue', 'disagreed':'green'}

print(data['nounScore'].max())
for cat in cats:
    d = data[data.label == cat]

    plt.hist(d['nounScore'], color=colors[cat], alpha = alphas[cat], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.show()
'''

thresh = [0.5 + t * 0.01 for t in range(5)]
correct = {}
for t in thresh:
    correct[t] = 0

c = 0
for _, row in data.iterrows():
    for t in thresh:
        if row['nounScore'] < t:
            predict = 'unrelated'
        else:
            predict = 'agreed'

        if row['label'] == predict:
            correct[t] += 1

    c += 1
    pCount = c * 100 / data.shape[0]
    print(f"Calculating Correct Rate: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 
    
print(' ' * 140, end='\r')
for t in thresh:
    print(f"Correct rate for {t} is {correct[t] / c}")


