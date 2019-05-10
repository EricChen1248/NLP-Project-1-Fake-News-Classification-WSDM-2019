import csv
import jieba

print("====== Cutting Words =======")

# jieba.enable_parallel(8)
# jieba.load_userdict('dict.txt.big')
stopWordFile = './0-Dictionaries/stopwords.txt'

stopWords = []
with open(stopWordFile) as f:
    for w in f:
        stopWords.append(w[:-1])
stopWords = set(stopWords)

def fileLen(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i

files = ["./data/train.csv", "./data/test.csv"]
dest = ["./data/cut_training.csv", "./data/cut_test.csv"]
for f, d in zip(files, dest):
    length = fileLen(f)

    count = 0
    # id,tid1,tid2,title1_zh,title2_zh,title1_en,title2_en,label
    with open(f, 'r') as csvFile:
        with open(d, 'w', newline='') as output:
            reader = csv.DictReader(csvFile, delimiter=',')
            writer = csv.DictWriter(output, ['id', 'id1', 'id2', 'title1', 'title2'])
            writer.writeheader()
            for row in reader:
                count += 1
                w = writer.writerow(
                {
                    'id' : row['id'],
                    'id1' : row['tid1'],
                    'id2' : row['tid2'],
                    'title1' : '/'.join([word for word in filter(lambda x: x not in stopWords, filter(None, jieba.cut(row['title1_zh'].replace('\n', ''), cut_all=False)))]),
                    'title2' : '/'.join([word for word in filter(lambda x: x not in stopWords, filter(None, jieba.cut(row['title2_zh'].replace('\n', ''), cut_all=False)))]),
                })

                pCount = count * 100 / length
                print(f"Cutting words: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

    print(' ' * 140)