import pandas as pd


def GenSentDict() -> tuple:
    '''
    sentDoc = pd.read_csv('../SentimentDictionary/sentiment_XS_30k.txt', delimiter = ',')

    pos = sentDoc[sentDoc.labels == 'positive']
    posWords = []
    for _, row in pos.iterrows():
        posWords += row['text'].split(' ')

    neg = sentDoc[sentDoc.labels == 'negative']
    negWords = []
    for _, row in neg.iterrows():
        negWords += row['text'].split(' ')

    '''

    posWords = [word[:-1] for word in open('../SentimentDictionary/ntusd-positive.txt', 'r')]
    negWords = [word[:-1] for word in open('../SentimentDictionary/ntusd-negative.txt', 'r')]
    return (set(posWords), set(negWords))

sentDict = GenSentDict()

def GetSentiment(tokens, sentDict):
    score = 0
    countword = 0
    for w in tokens:
        if w in sentDict[0]:
            score += 1
            countword += 1
        elif w in sentDict[1]:
            score -= 1
            countword += 1

    if countword != 0:
        return score/countword
    else:
        return 0


data = pd.read_csv('../cutTrainingData.csv', delimiter = ',')
length = data.shape[0]
score = []
c = 0
for index, row in data.iterrows():
    try:
        score.append(GetSentiment(row['title1'].split('/'), sentDict) - GetSentiment(row['title2'].split('/'), sentDict))
    except Exception:
        score.append(0)
    
    c += 1
    pCount = c * 100 / length
    print(f"Calculating Score: ({'#' * int(pCount)}{'-' * (100 - int(pCount))}) {pCount:.2f}%", end='\r') 

print(' ' * 140)

data['sentiment'] = score
data = data.drop('title1', axis = 1)
data = data.drop('title2', axis = 1)

data.to_csv('../data/sentimentScore.csv', index = None, header = True)