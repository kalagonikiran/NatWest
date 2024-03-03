step 1:Downloaded the Testdata.csv dataset and read the file using pandas and coverted the file to a dataframe

step 2:Checked the data types of columns

step 3:checked the target column and converted the Acknowledged column to 1 and rest all to 0

step 4:Filled the missing price values with median of the column since there are significant outliers and dropped the rows of remaing columns with null values

step 5:By using OneHotEncoder categorical columns are coverted to numerical values

step 6:Split the data into x and y variables where y is the target column

step 7:x and y are splitted into training and testing data

step 8:The training data is tested against RandomForestClassifier

step 9:Model is evaluated using various metrics like roc_curve,roc_auc_score_value,confusion_matrix and visualisations are made based on metrics

step 10:Calssification report is made.
