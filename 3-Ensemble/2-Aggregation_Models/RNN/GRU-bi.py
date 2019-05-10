import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import keras
from keras.utils import to_categorical
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
# Any results you write to the current directory are saved as output.
from keras import Input
from keras.layers import Embedding, LSTM, concatenate, Dense, GRU, Bidirectional
from keras.models import Model
from keras.optimizers import Adam, Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint

TRAIN_CSV_PATH = './data/cut_training.csv.csv'
TEST_CSV_PATH = './data/cut_testing.csv.csv'
TRAIN_LABEL_PATH = './data/label.npy'


MAX_NUM_WORDS = 100000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 100
DROPOUT = 0.4
EPOCHS = 5
BATCH = 512
NUM_CLASSES = 3


def load_data():
    # read features from csv file and label from npy
    df_trainX = pd.read_csv(TRAIN_CSV_PATH)
    df_testX = pd.read_csv(TEST_CSV_PATH)
    df_trainX.title2.fillna('UNKNOWN', inplace=True)
    df_testX.title2.fillna('UNKNOWN', inplace=True)
    print(df_testX.isnull().any())
    Y_train = np.load(TRAIN_LABEL_PATH)
    # transfer data type to ndarray or reshape data
    X_train = df_trainX.as_matrix()
    X_train = X_train[:,1:3]
    X_test = df_testX.as_matrix()
    X_test = X_test[:, 3:]
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
    # print(X_train, X_test)
    # split validation data
    # X_all, X_train, X_valid = X_train, X_train[:-32000], X_train[-32000:]
    # Y_all, Y_train, Y_valid = Y_train, Y_train[:-32000], Y_train[-32000:]
    return X_train, Y_train, X_test

def data_preprocess():
    tokenizer = Tokenizer(num_words = MAX_NUM_WORDS, split='/')
    corpus = np.concatenate((X_all, X_test), axis=0)
    print(corpus.shape)
    corpus = np.reshape(corpus, (corpus.shape[0]*2,))
    
    tokenizer.fit_on_texts(corpus)
    # map sentences to sequences of numbers (word index)
    train_title1 = tokenizer.texts_to_sequences(X_all[:,0])
    train_title2 = tokenizer.texts_to_sequences(X_all[:,1])
    test_title1 = tokenizer.texts_to_sequences(X_test[:,0])
    test_title2 = tokenizer.texts_to_sequences(X_test[:,1])
    # Zero Padding
    train_title1 = pad_sequences(train_title1, maxlen=MAX_SEQUENCE_LENGTH)
    train_title2 = pad_sequences(train_title2, maxlen=MAX_SEQUENCE_LENGTH)
    test_title1 = pad_sequences(test_title1, maxlen=MAX_SEQUENCE_LENGTH)
    test_title2 = pad_sequences(test_title2, maxlen=MAX_SEQUENCE_LENGTH)
    # Label One Hot Encoding
    label = to_categorical(Y_all)
    print(label[0:5,:])
    # split validation data
    train_title1, valid_title1 = train_title1[:-32000], train_title1[-32000:]
    train_title2, valid_title2 = train_title2[:-32000], train_title2[-32000:]
    train_label, valid_label = label[:-32000], label[-32000:]
    return train_title1, valid_title1, test_title1, train_title2, valid_title2, test_title2, train_label, valid_label

def train():
    # Input Layer
    top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    bm_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    # Embedding Layer
    embedding_layer = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM)
    top_embedded = embedding_layer(top_input)
    bm_embedded = embedding_layer(bm_input)
    # LSTM Layer
    shared_lstm = Bidirectional(GRU(128))
    top_output = shared_lstm(top_embedded)
    bm_output = shared_lstm(bm_embedded)
    # concatenate
    merged = concatenate([top_output, bm_output], axis=-1)
    dense =  Dense(units=NUM_CLASSES, activation='softmax')
    predictions = dense(merged)
    model = Model(inputs=[top_input, bm_input], outputs=predictions)
    model.summary()

    #Build Model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	#Early Stop
    callbacks = []
    callbacks.append(ModelCheckpoint('./RNN/model-GRUbi/{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=False, period=1))
    model.summary()
    history = model.fit([train_title1, train_title2], train_label, validation_data=([valid_title1, valid_title2], valid_label), epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks)

    return


MODEL_PATH = './RNN/model-GRUbi/00001-0.74019.h5'
OUTPUT_FILE_PATH = './RNN/GRUbi.csv'
def predict():
    model = load_model(MODEL_PATH)
    result = model.predict([test_title1, test_title2], batch_size=512)
    output = [['Id', 'Category']]
    index_to_label = { 0: 'unrelated', 1: 'agreed', 2: 'disagreed'}
    df_id = pd.read_csv("./test.csv")
    df_id = df_id["id"]
    np_id = df_id.as_matrix()
    category = [index_to_label[idx] for idx in np.argmax(result, axis=1)]
    for i in range(np_id.shape[0]):
        output.append([np_id[i], category[i]])
    
    output = pd.DataFrame(output)
    output.to_csv(OUTPUT_FILE_PATH, index=False, header=False)


    
    


X_all, Y_all, X_test = load_data()
train_title1, valid_title1, test_title1, train_title2, valid_title2, test_title2, train_label, valid_label = data_preprocess()
# train()
predict()

