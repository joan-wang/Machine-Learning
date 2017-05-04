import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from IPython.display import display

def explore(df):
    '''
    Explore the data and print tables to TEXT_OUTPUT file.
    '''
    if TERMINAL: 
        orig_stdout = sys.stdout
        f = open(TEXT_OUTPUT, 'w')
        sys.stdout = f

        print("Shape: \n", df.shape, '\n')
        print("Data types: \n", df.dtypes, '\n')
        print("Summary statistics: \n", df.describe(), '\n')
        print("Percent null per column: \n", df.isnull().sum()/df.shape[0], '\n')
        print("Proportion of", OUTCOME_VAR, ': \n', df[OUTCOME_VAR].value_counts(normalize=True), '\n')
    else:
        display("Shape:", df.shape)
        display("Data types:", df.dtypes)
        display("Summary statistics:", df.describe())
        display("Percent null per column:", df.isnull().sum()/df.shape[0])
        display("Proportion of", OUTCOME_VAR, df[OUTCOME_VAR].value_counts(normalize=True))

    if TERMINAL:
        sys.stdout = orig_stdout
        f.close()

    graph(df)

def graph(df):   
    '''
    This function contains graphs specific to the credit data dataset
    '''
    
    # Plot histograms of the six "number of" variables
    fig = plt.figure(figsize=(10,30))
    num = 611
    for i in ['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents']:
        ax = plt.subplot(num)
        df[i].plot.hist(bins=50)
        ax.set_xlabel(i)
        ax.set_ylabel("Distribution")
        ax.set_title('Histogram of '+ i)
        num +=  1

    sns.plt.show()

    # Plot histograms with limited range for three of the six "number of" variables
    # Histograms are cut off at 5 because of extremely long right tail. 
    fig = plt.figure(figsize=(10,15))
    num = 311
    for i in ['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse']:
        ax = plt.subplot(num)
        df[i].plot.hist(bins=100)
        ax.set_xlim((0,5))
        ax.set_xlabel(i)
        ax.set_ylabel("Distribution")
        ax.set_title('Limited range distribution of '+ i)
        num += 1
    sns.plt.show()
    
    # Boxplot of "Number of times" variables
    bx = df.boxplot(column=['NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'], rot=90)
    bx.set_xlabel("Variables")
    bx.set_ylabel("Distribution")
    bx.set_title('Distribution of "Number of" Variables')
    bx.set_ylim([0, 22])
    sns.plt.show()
    
    # Histogram of age distribution
    cx = df['age'].plot.hist(bins=10)
    cx.set_xlabel("Age")
    cx.set_ylabel("Distribution")
    cx.set_title("Distribution of Age") 

    # Histogram of income distribution
    cx = df['MonthlyIncome'].plot.hist(bins=200)
    cx.set_xlim((0,100000))
    cx.set_xlabel("Monthly Income")
    cx.set_ylabel("Distribution")
    cx.set_title("Limited range distribution of Monthly Income") 

    # Correlation matrix of all the variables
    # Source: http://seaborn.pydata.org/examples/many_pairwise_correlations.html
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    ax.set_title('Heat map of correlation between variables')
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True,
             cbar_kws={"shrink": .5}, ax=ax)
    sns.plt.show()

    # Histogram of age NOT WORKING!!
    #pd.value_counts(df.groupby('age').plot(kind='bar'))
    '''
    f, ax = plt.hist(df['age'], bins=10)
    ax.set_xlabel("Age")
    ax.set_ylabel("Distribution")
    ax.set_title('Histogram of age')
    '''
    
   
    
