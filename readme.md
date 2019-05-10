# Fake News Classification for 2019 WSDM Fake News Classification on Kaggle
![outline](https://github.com/EricChen1248/NLP-Project-1-Fake-News-Detection/blob/master/images/outline.png)

[kaggle](https://www.kaggle.com/c/fake-news-pair-classification-challenge/overview)

[paper](https://github.com/EricChen1248/NLP-Project-1-Fake-News-Detection/blob/master/report.pdf)

## Requirements
Python Version: Python 3.6+

Install Required Packages:
````
pip install -r requirements.txt
````

## Input
To run this program, need to put `train.csv` and `test.csv` in ./data

## Execution
Execute the following python scripts in numeric order

````bash
|-- 1-Preprocessing:
|   |-- cutting words(Chinese and English)、word2vec embedding
|-- 2-K-means:
|   |-- 1-calculate_sentence_vector.py: 		# sentence vector
|   |-- 2-k_means_on_sentence_vectors.py: 		# K-means scoring on sentence vector
|-- 3-Ensemble:
|   |-- 1-Calculate_Features:
|   |-- |-- 1-calc_sentence_embedding.py: 		# sentence embedding
|   |-- |-- 2-calc_english_noun_similarity.py: 		# calculate English noum similarity
|   |-- |-- 3-calc_sentiment_score.py: 			# calculate sentiment score
|   |-- |-- 4-string_matching.py: 			# calculate string matching ratio
|   |-- |-- 5-get_label.py: 				# transform category labels to integers
|   |-- |-- 6-combine.py: 				# combine all the features in a csv file
````

Run the chosen aggregation models

```` bash
Tree based:
classifier.py
	used models:			# predicting testing data, generate `submission.csv`
		decision tree
		random forest
		xgboost
	needed files: 
		feature_train.csv: 	# training data feature
		feature_test.csv: 	# testing data feature
		label.npy: 		# training data label
	used features:
		overlap_ratio: 		# overlap ratio of two titles
		partial_ratio: 		# partial overlap ratio of two titles
		tokenset_ratio: 	# token set ratio of two titles
		rumor: 			# word '谣' '官方' '假' '真相'
		EnglishNounScore: 	# similarity of English noun
		EmbeddingScore: 	# sentence embedding
	kmeans: 			# scoring of K-means

RNN based:

used models:GRU-1 GRU-bi GRU-multi GRU-multi-2 GRU-w2v LSTM-1 LSTM-biLSTM-w2v
needed files: 
	cut_training.csv: 		# cutted training data
	cut_testing.csv: 		# cutted testing data
	label.npy: 			# training data label
	word2vec_model
````
````bash
9-Tools:
  compareCorrect.py: 		# calculate K-means accuracy
  testing_embedding_model.py: 	# calculate cosine similarity accuracy
  visualize_english.py: 	# calculate English noun similarity accuracy
  Visualizer.py: 		# visualization
````

## Submission
The generated `submission.csv` is the result that can be submitted into kaggle.

## Citations and References

````bibtex
@ONLINE{wsdm,
	URL = "www.kaggle.com/c/fake-news-pair-classification-challenge/overview",
	AUTHOR = "{WSDM}",
	TITLE = "{WSDM} - Fake News Classification",
	YEAR = "2019",
}
@ONLINE{jieba,
	URL = "github.com/fxsjy/jieba",
	AUTHOR = "fxsjy",
	TITLE = "Jieba Github Repository",
	YEAR = "2018",
}
@ONLINE{nltk,
	URL = "www.nltk.org/api/nltk.tokenize.html",
	AUTHOR = "Steven Bird, Edward Loper, Ewan Klein",
	TITLE = "Natural Language Toolkit",
	YEAR = "2018",
}

@ONLINE{gensim,
	URL = "radimrehurek.com/gensim/models/word2vec.html",
	AUTHOR = "Radim Řehůřek",
	TITLE = "Gensim Model",
	YEAR = "2018",
}

@ONLINE{percTag,
	URL = "www.nltk.org/_modules/nltk/tag/perceptron.html",
	AUTHOR = "Steven Bird, Edward Loper, Ewan Klein",
	TITLE = "{NLTK - Perceptron Tagger}",
	YEAR = "2018",
}
````