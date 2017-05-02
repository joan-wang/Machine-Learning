import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from config import *

def explore(df):
    '''
    Explore the data and print tables to TEXT_OUTPUT file.
    '''
    orig_stdout = sys.stdout
    f = open(TEXT_OUTPUT, 'w')
    sys.stdout = f

    print("Shape: \n", df.shape, '\n')
    print("Data types: \n", df.dtypes, '\n')
    print("Summary statistics: \n", df.describe(), '\n')
    print("Percent null per column: \n", df.isnull().sum()/df.shape[0], '\n')
    print("Proportion of", OUTCOME_VAR, ': \n', df[OUTCOME_VAR].value_counts(normalize=True), '\n')
    sys.stdout = orig_stdout
    f.close()

    graph(df)

def graph(df):
    
    fig = plt.figure(figsize=(11,13))
    
    ax = df.boxplot(column=['NumberOfTime30-59DaysPastDueNotWorse','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents'], rot=90)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Distribution")

    # NEED MORE PLOTS
    '''
    num = 321
    for col in match_scores:
        plt.subplot(num)
        where, part, what = col.partition('_')
        plt.title(what + ' from ' + where)
        plt.xlabel('Jaro-Winkler score')
        match_scores[col].hist(grid=False)
        num = num + 2

    num = 322
    for col in unmatch_scores:
        plt.subplot(num)
        where, part, what = col.partition('_')
        plt.title(what + ' from ' + where)
        plt.xlabel('Jaro-Winkler score')
        unmatch_scores[col].hist(grid=False)
        num = num + 2
   
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.9, wspace=0.35)
    '''
    fig.savefig(GRAPH_OUTPUT)
