import sys
import pandas as pd
import numpy as np
from config import *
from munging import *
from model import *
from features import *
from explore import *
from sklearn.cross_validation import train_test_split



'''
Notes and to dos
- make labeled graphs
- classifiers: Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, Random Forests, Boosting, and Bagging
- 1-2 page writeup
- Categorical variables not ok for model? "could not convert string to float"
- need to fix feature selection config file; currently hard coded
- where to display results? store more in results to graph and evaluate?

'''

def pipeline(df):
    explore(df)
    df = clean(df)
    X_train, X_test, y_train, y_test = train_test_split(df.ix[:, 1:], df[OUTCOME_VAR], test_size=TEST_SIZE, random_state=0)
    X_train = feature_eng(X_train)
    X_test = feature_eng(X_test)
    results = classifiers_loop(X_train, X_test, y_train, y_test)
    return results

if __name__=="__main__":
    num_args = len(sys.argv)

    if num_args != 2:
        print("usage: python3 " + sys.argv[0] + "<csv file name>")
        sys.exit(0)
  
    df = pd.read_csv(sys.argv[1], index_col=INDEX_COL, sep=SEPERATOR)
    restuls = pipeline(df)