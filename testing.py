import csv
from gensim.models import Word2Vec

print("====== Testing Threshold =======")

def fileLen(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i
length = fileLen('cutTrainingData.csv') - 290_000

model = Word2Vec.load("word2vec_model")

start = 4
best = 0
bestThresh = 0

scores = []
with open('cutTrainingData.csv', 'r') as csvFile:
    reader = csv.DictReader(csvFile, delimiter=',')
    count = {'unrelated': 0, 'agreed': 0, 'disagreed': 0, 'unrelatedCorrect': 0, 'relatedCorrect': 0}
    for _ in range(290_000):
        next(reader)

    c = 0
    for row in reader:
        label = row['label']
        count[label] += 1

        score = model.wv.wmdistance(row['title1'].split('/'), row['title2'].split('/'))
        scores.append((score, label))
        
        c += 1
        pCount = c * 100 / length
        print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

    print(' ' * 140)            

for i in range(80):
    count['unrelatedCorrect'] = 0
    count['relatedCorrect'] = 0

    threshold = start + 0.1 * i

    for score, label in scores:
        if score > threshold:
            if label == 'unrelated':
                count['unrelatedCorrect'] += 1
        else:
            if label == 'agreed':
                count['relatedCorrect'] += 1
                
    accuracy = (count['unrelatedCorrect'] + count['relatedCorrect']) / (count['unrelated'] + count['agreed'] + count['disagreed'])
    print(f"{threshold} : {accuracy}")
    print(count)

    if accuracy > best:
        best = accuracy
        bestThresh = threshold

print(f"Best Threshold: {bestThresh}, with accuracy of {best}")