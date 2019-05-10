import csv
from gensim.models import Word2Vec

print("====== Training Word2Vec =======")

count = 0

titles = []
with open('./data/cut_training.csv', 'r') as csvFile:
    reader = csv.DictReader(csvFile, delimiter=',')
    for row in reader:
        titles.append(row['title1'].split('/'))
        titles.append(row['title2'].split('/'))
        
model = Word2Vec(titles, size=70, window=2, min_count=10, workers=16)
model.save("./data/word2vec_model")