import numpy
import pandas

data = pandas.read_csv('./data/train.csv')
results = numpy.load('./data/k-means-score.npy')

resultMap = {1:'agreed', -1:'disagreed', 0:'unrelated'}

data['result'] = results

correctCount = {'agreed': 0, 'disagreed': 0, 'unrelated': 0}
for _, row in data.iterrows():
    if row['label'] == resultMap[row['result']]:
        correctCount[row['label']] += 1

    
print(correctCount)