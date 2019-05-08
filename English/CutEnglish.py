import nltk
import pandas
import contractions
from nltk.tokenize import word_tokenize
import json

data = pandas.read_csv('./data/train.csv')
stop_words = set(nltk.corpus.stopwords.words('english')) 

text = {}
titles = [['tid1', 'title1_en'], ['tid2', 'title2_en']]
c = 0
for _, row in data.iterrows():
    for id, title in titles:
        if row[id] not in text:
            sent = word_tokenize(contractions.fix(row[title].lower()))
            filtered = [w for w in sent if w not in stop_words]
            text[row[id]] = [w for w in filtered]
        
    
    c += 1
    pCount = c * 100 / data.shape[0]
    print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 
            
print(' ' * 140)
cutFile = './cutEnglish.json'
with open(cutFile, 'w') as f:
    json.dump(text, f)
