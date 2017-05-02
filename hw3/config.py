# title of index column to use when importing csv
INDEX_COL = 'PersonID'

# name of output variable to be predicted (values should be 0/1)
OUTCOME_VAR = 'SeriousDlqin2yrs'

# preliminary list of features to use
FEATURES = ['RevolvingUtilizationOfUnsecuredLines',
    'age',                                       
    'zipcode',                                   
    'NumberOfTime30-59DaysPastDueNotWorse',      
    'DebtRatio',                               
    'MonthlyIncome',                           
    'NumberOfOpenCreditLinesAndLoans',           
    'NumberOfTimes90DaysLate',                   
    'NumberRealEstateLoansOrLines',              
    'NumberOfTime60-89DaysPastDueNotWorse',      
    'NumberOfDependents']

# separator character for csv
SEPERATOR = ','

# names of file to print explore.py text and graphs
TEXT_OUTPUT = 'explore.txt'
GRAPH_OUTPUT = 'graphs.pdf'

# percent split for training vs teseting
TEST_SIZE = .20

# columns that need versions
TO_INT = ['NumberOfDependents']
TO_FLOAT = []
TO_STR = []
TO_BOOL = ['SeriousDlqin2yrs']

# columns with extreme values that need capping
EXTREME_COLS = ['RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']

# percentile value at which to cap EXTREME_COLS
CAP = 0.999

# continuous variables that need discretizing
BUCKETING_COLS = ['RevolvingUtilizationOfUnsecuredLines','DebtRatio']

# number of buckets to cut the BUCKETING_COLS into
Q = 10

# categorical variables to be turned into dummies; not applicable for credit data variables
CATEGORICAL = []