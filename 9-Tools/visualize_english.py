import pandas
import numpy
import matplotlib.pyplot as plt


dataFile = ['./data/train.csv', './data/test.csv']
nounFile = ['./english_noun_score.', './data/english_noun_score_test.npy']

for d, n in zip(dataFile, nounFile):
    data = pandas.read_csv(d)
    nouns = numpy.load(n)

    data['nounScore'] = nouns

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


