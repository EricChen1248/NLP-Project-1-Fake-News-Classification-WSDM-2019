import pandas

combined = pandas.read_csv('englishNounTest.csv')
embed = pandas.read_csv('./data/embeddingScoreTest2.csv')
#hsiao = pandas.read_csv('feature_train.csv')

combined['EmbeddingScore'] = embed['embedding']
combined = combined.round(5)

combined.to_csv('./combined.csv', index = False)