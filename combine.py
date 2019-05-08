import pandas

combined = pandas.read_csv('englishNoun.csv')
embed = pandas.read_csv('./data/embeddingScore.csv')
hsiao = pandas.read_csv('feature_train.csv')

combined['EmbeddingScore'] = embed['score']
combined = combined.round(5)
combined.to_csv('./combined.csv', index = False)