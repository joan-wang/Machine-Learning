Assignment 2: Machine Learning Pipeline

Goal: The goal of this assignment is to build a simple, modular, extensible, machine learning pipeline in Python. 
The pipeline should have functions that can do the following tasks:
1. Read/Load Data
2. Explore Data
3. Pre-Process and Clean Data
4. Generate Features/Predictors
5. Build Machine Learning Classifier
6. Evaluate Classifier

Problem: Predict who will experience financial distress in the next two years.
Outcome variable: SeriousDlqin2yrs

Note: Focus of this assignment was on the pipeline and not the fit of the model or solving the problem. 
No validation or cross-validation was used. 

Overview of work product:
- Read csv into a Pandas dataframe
- Explored the data (data types, descriptive statistics, distributions)
- Filled in NAs with median/mean
- Addressed skew in data. Many of the variables have long right tails (extreme upper outliers). I capped these columns at a 99.9% percentile ceiling to address outliers without removing them. 
- Bucketed continuous variables ('RevolvingUtilizationOfUnsecuredLines','DebtRatio') into discrete bins, one for each 10% decile. Bucket ranges are variable but sizes are approximately equal (same number of records in each). 
- Turned categorical variables ("NumberOfDependents", "age") into additional columns of dummy variables. 
- Chose features to include in logistic regression. Did not include age bins because they decreased the model performance.
- Built a logistic regression model for the available data. As indicated in instructions, I did not separate out a training and test set or use a rigorous process for selecting features. 
- Evaluated the model using basic accuracy score. The model accuracy score is 93.37%. The percentage of individuals in the dataset with SeriouDlqin2yrs=1 is 6.7%. Thus, by trivially guessing 0 for each record, the baseline level of accuracy would be 93.33%. This model performs only slightly better than random guessing.
- Using classification_report, we see that precision is 0.6 and recall is 0.03.
- For this type of analysis, we should be more concerned about false negatives than false positives. Assuming that the policy action resulting from this analysis involves intervening with individuals likely to experience financial distressuccs, we would rather classify more people as likely to experience financial distress than there really is, rather than allow people to fall through the cracks. In other words, we should restrict the number of false negatives at the expense of allowing more false positives.
- As a next step (beyond this assignment), I would improve the model by better dealing with outliers, focusing on improved feature selection, trying out multiple classifiers, and further exploring the precision-recall curve to satisfy the policy goal discussed above.
