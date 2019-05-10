import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import jieba.posseg as pseg
import os
import keras
from sklearn.tree import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import random
from fuzzywuzzy import fuzz
import xgboost as xgb

TRAIN_LABEL_PATH = "./data/label.npy" 
TRAIN_FEATURE_PATH = "./data/feature_train.csv"
TEST_FEATURE_PATH = "./data/feature_test.csv"



def load_data():
    # read features from csv file and label from npy
    df_trainX = pd.read_csv(TRAIN_FEATURE_PATH,index_col=False)
    df_testX = pd.read_csv(TEST_FEATURE_PATH)
    print(df_trainX.head())
    print(df_testX.head())
    Y_train = np.load(TRAIN_LABEL_PATH)
    # transfer data type to ndarray or reshape data
    X_train = df_trainX.as_matrix()
    X_train = X_train[:,1:]
    X_test = df_testX.as_matrix()
    X_test = X_test[:,1:]
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
    # split validation data
    X_all, X_train, X_valid = X_train, X_train[:-32000], X_train[-32000:]
    Y_all, Y_train, Y_valid = Y_train, Y_train[:-32000], Y_train[-32000:]
    return X_all, X_train, X_valid, Y_all, Y_train, Y_valid, X_test

def predictor(Y_test, file_path):
    result = []
    for res in Y_test:
        if res == 0:
            result.append("unrelated")
        elif res == 1:
            result.append("agreed")
        else:
            result.append("disagreed")
    df_id = pd.read_csv("./data/test.csv")
    df_id = df_id["id"]
    np_id = df_id.as_matrix()
    np_id = np.reshape(np_id, (np_id.shape[0], 1))
    result = np.array(result)
    result = np.reshape(result, (result.shape[0], 1))
    out = np.concatenate((np_id, result), axis=1)
    out = pd.DataFrame(out, columns=["Id", "Category"])
    out.to_csv(file_path, index=False)

def decision_tree():
    decision_tree_clf = tree.DecisionTreeClassifier()
    decision_tree_clf = decision_tree_clf.fit(X_train, Y_train)
    Y_valid_pred = decision_tree_clf.predict(X_valid)
    acc = accuracy_score(Y_valid, Y_valid_pred)
    decision_tree_clf = decision_tree_clf.fit(X_all, Y_all)
    Y_test = decision_tree_clf.predict(X_test)
    predictor(Y_test, "decision_tree_all.csv")
    print('decision_tree_validation: ',acc)

def random_forest():
    random_forest_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    random_forest_clf = random_forest_clf.fit(X_train, Y_train)
    Y_valid_pred = random_forest_clf.predict(X_valid)
    acc = accuracy_score(Y_valid, Y_valid_pred)
    random_forest_clf = random_forest_clf.fit(X_all, Y_all)
    Y_test = random_forest_clf.predict(X_test)
    predictor(Y_test, "random_forest_all.csv")
    print('random_forest_validation: ',acc)

def xgboost():
    xgboost_clf = XGBClassifier()
    xgboost_clf = xgboost_clf.fit(X_train, Y_train)
    Y_valid_pred = xgboost_clf.predict(X_valid)
    acc = accuracy_score(Y_valid, Y_valid_pred)
    xgboost_clf = xgboost_clf.fit(X_all, Y_all)
    Y_test = xgboost_clf.predict(X_test)
    #predictor(Y_test, "xgboost_all.csv")
    predictor(Y_test, "submission_25.csv")
    print('xgboost_validation: ',acc)

    


X_all, X_train, X_valid, Y_all, Y_train, Y_valid, X_test = load_data()
decision_tree()
random_forest()
xgboost()
