import pandas as pd
import numpy as np
from config import *

def feature_eng(df):
    df = fill_missing(df)
    df = cap_extreme(df)
    df = discretize(df)
    df = make_dummies(df)
    return df

def fill_missing(df):
    '''
    Function to fill null values in df with:
        median (if integer)
        mean (if float)
        mode (if string)
    Do this imputation after training/test split occurs.
    '''   
    for colname in df:    
        if 'int' in str(df[colname].dtype):
            df[colname].fillna(value=df[colname].median(), inplace=True)
        elif 'float' in str(df[colname].dtype):
            df[colname].fillna(value=df[colname].mean(), inplace=True)
        elif df[colname].dtype == 'object':
            try: # see if mode exists
                mode = df[colname].mode()[0]
                df[colname].fillna(value=mode, inplace=True)
            except: # if no mode, fill with 'unknown'
                df[colname].fillna(value='Unknown', inplace=True)
        else:
            sys.exit('check irregular data types')
    return df

def cap_extreme(df):
    '''
    Function to cap extreme columns at a certain percentile
    '''
    for c in EXTREME_COLS:
        ceiling = df[c].quantile(CAP)
        df[c] = df[c].apply(lambda x: set_ceiling(x, ceiling))
    return df

def set_ceiling(x, ceiling):
    '''
    Helper funciton for cap_extreme
    '''
    if x > ceiling:
        return ceiling
    else:
        return x

def discretize(df):
    '''
    Function for discretizing continuous variables into Q equally-sized buckets     
    '''
    for c in BUCKETING_COLS:
        df['bins_' + c] = pd.qcut(df[c], q=Q)
    return df

def make_dummies(df):
    '''
    Function to make dummy features from categorical variables and concatenate with df
    '''
    for c in CATEGORICAL:
        dummies = pd.pd.get_dummies(df[i], prefix=i)
        df = pd.concat([df, dummies], axis=1)
    return df