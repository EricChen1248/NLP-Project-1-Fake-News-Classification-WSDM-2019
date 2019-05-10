required Python3 
required packages: pip install -r requirements.txt
To run this program, need to put `train.csv` and `test.csv` in data/
then run each .py file in filename order.

0-Dictionaries:
Chinese dictionary、sentiment dictionary、stopwords

1-Preprocessing:
cutting words(Chinese and English)、word2vec embedding

2-K-means:
  1-calculate_sentence_vector.py: sentence vector
  2-k_means_on_sentence_vectors.py: K-means scoring on sentence vector

3-Ensemble:
  1-Calculate_Features:
    1-calc_sentence_embedding.py: sentence embedding
    2-2-calc_english_noun_similarity.py: calculate English noum similarity
    3-3-calc_sentiment_score.py: calculate sentiment score
    4-string_matching.py: calculate string matching ratio
    5-get_label.py: transform category labels to integers
    6-combine.py: combine all the features in a csv file
  2-Aggregation_Models:
  tree based:
	classifier.py:
		used models: decision tree、random forest、xgboost
		predicting testing data, generate `submission.csv`
	needed files: 
		feature_train.csv: training data feature
		feature_test.csv: testing data feature
		label.npy: training data label
	used features:
  		overlap_ratio: overlap ratio of two titles
  		partial_ratio: partial overlap ratio of two titles
  		tokenset_ratio: token set ratio of two titles
  		rumor: word '谣' '官方' '假' '真相'
  		EnglishNounScore: similarity of English noun
  		EmbeddingScore: sentence embedding
  		kmeans: scoring of K-means
  RNN based:
	used models:GRU-1 GRU-bi GRU-multi GRU-multi-2 GRU-w2v LSTM-1 LSTM-bi LSTM-w2v
	needed files: 
		cut_training.csv: cutted training data
		cut_testing.csv: cutted testing data
		label.npy: training data label
		word2vec_model

9-Tools:
  compareCorrect.py: calculate K-means accuracy
  testing_embedding_model.py: calculate cosine similarity accuracy
  visualize_english.py: calculate English noun similarity accuracy
  Visualizer.py: visualization
