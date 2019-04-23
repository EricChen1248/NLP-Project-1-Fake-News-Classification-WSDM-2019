import csv
import jieba

print("====== Cutting Words =======")


# jieba.enable_parallel(8)
# jieba.load_userdict('dict.txt.big')

stopWords = []
with open('stopwords.txt') as f:
    for w in f:
        stopWords.append(w[:-1])
stopWords = set(stopWords)
'''

jieba.analyse.set_stop_words('stopwords.txt')
'''


def fileLen(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i
length = fileLen('data/train.csv')

count = 0
# id,tid1,tid2,title1_zh,title2_zh,title1_en,title2_en,label
with open('data/train.csv', 'r') as csvFile:
    with open('cutTrainingData.csv', 'w', newline='') as output:
        reader = csv.DictReader(csvFile, delimiter=',')
        writer = csv.DictWriter(output, ['id', 'title1', 'title2', 'label'])
        writer.writeheader()
        for row in reader:
            count += 1
            w = writer.writerow(
            {
                'id' : row['id'],
                'title1' : '/'.join([word for word in filter(lambda x: x not in stopWords, filter(None, jieba.cut(row['title1_zh'].replace('\n', ''), cut_all=False)))]),
                'title2' : '/'.join([word for word in filter(lambda x: x not in stopWords, filter(None, jieba.cut(row['title2_zh'].replace('\n', ''), cut_all=False)))]),
                'label' : row['label']
            })

            pCount = count * 100 / length
            print(f"Cutting words: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 
