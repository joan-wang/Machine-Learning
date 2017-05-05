import sys
import pandas as pd
import numpy as np
from config import *
from cleaning import *
from model import *
from features import *
from explore import *
from sklearn.cross_validation import train_test_split


def pipeline(df):
    explore(df)
    df = clean(df)
    if len(FEATURE_COLS)>0:
        # use feature names specified in FEATURE_COLS
        X_train, X_test, y_train, y_test = train_test_split(df[FEATURE_COLS], df[OUTCOME_VAR], test_size=TEST_SIZE, random_state=0)
    else: 
        # use all features in dataframe
        X_train, X_test, y_train, y_test = train_test_split(df.ix[:, 1:], df[OUTCOME_VAR], test_size=TEST_SIZE, random_state=0)
    X_train = feature_eng(X_train)
    X_test = feature_eng(X_test)
    results = classifiers_loop(X_train, X_test, y_train, y_test)
    results.to_csv('results.csv')
    return results, y_test

if __name__=="__main__":
    num_args = len(sys.argv)

    if num_args != 2:
        print("usage: python3 " + sys.argv[0] + "<csv file name>")
        sys.exit(0)
  
    df = pd.read_csv(sys.argv[1], index_col=INDEX_COL, sep=SEPERATOR)
    results, y_test = pipeline(df)
    