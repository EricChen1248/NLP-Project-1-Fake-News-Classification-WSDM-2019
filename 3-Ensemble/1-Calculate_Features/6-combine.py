import pandas as pd
import numpy as np


string = pd.read_csv('./data/string_train.csv')
embed = np.load('./data/embedding_train.npy')
english = np.load("./data/english_noun_score.npy")
kmeans = np.load('./data/k-means-score.npy')
df = pd.DataFrame({'EnglishNounScore':english,'EmbeddingScore':embed,'kmeans':kmeans})
string['EnglishNounScore'] = df['EnglishNounScore']
string['EmbeddingScore'] = df['EmbeddingScore']
string['kmeans'] = df['kmeans']


string.to_csv('./feature_train.csv')

string = pd.read_csv('./data/string_test.csv')
embed = np.load('./data/embedding_test.npy')
english = np.load("./data/english_noun_score_test.npy")
kmeans = np.load('./data/k-means-score-test.npy')
df = pd.DataFrame({'EnglishNounScore':english,'EmbeddingScore':embed,'kmeans':kmeans})
string['EnglishNounScore'] = df['EnglishNounScore']
string['EmbeddingScore'] = df['EmbeddingScore']
string['kmeans'] = df['kmeans']


string.to_csv('./feature_test.csv')