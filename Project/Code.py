#first we need to import required libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# We want our plots to appear in the notebook
%matplotlib inline 

#readind the test data
df=pd.read_csv("Testdata.csv")
df.head(15)

df['Reporting Status'].value_counts()
#since we are considering Reporting Status as target column

df["Reporting Status"]=df["Reporting Status"].replace({"Acknowledged":1,"Failed Acknowledgement":0,"Error":0,"Ignored":0,"Processing Error":0,"ACK":1,"Failed Ack":0})
df.dtypes 

median_price=df["price"].median()
df["price"]=df["price"].fillna(median_price)
df.dropna(inplace=True)
df.isnull().sum()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
x=df.drop(["Reporting Status","clDateTime","cDateTime","expirationDate","Timestamp","quantity","endDate","terminationDate"],axis=1)
y=df["Reporting Status"]
# convert the categorical columns to one hot encoded
# Turn the categories into numbers
catagorical_features=["regulator","assetClass","cflag","method","eventT","mType","seller","sType","Product","party","transactionType","clStatus","eFlag"]
one_hot=OneHotEncoder()
transformer1=ColumnTransformer([("one_hot",one_hot,catagorical_features)],remainder='passthrough',sparse_threshold=0)
transformed_x=transformer1.fit_transform(x)
print(transformed_x)



#spliting data into training and testing data 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(transformed_x,y)
clf = RandomForestClassifier()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)


#Metrics to evaluate a model
from sklearn.metrics import roc_curve
y_probs = clf.predict_proba(x_test)
y_probs = y_probs[:, 1]
# Calculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#The maximum ROC AUC score you can achieve is 1.0 and generally, the closer to 1.0, the better the model.
from sklearn.metrics import roc_auc_score
roc_auc_score_value = roc_auc_score(y_test, y_probs)
print(roc_auc_score_value)

from sklearn.metrics import confusion_matrix
y_preds = clf.predict(x_test)
confusion_matrix(y_test, y_preds)
ConfusionMatrixDisplay.from_predictions(y_true=y_test, 
                                        y_pred=y_preds);


#corelation matrix
correlation_matrix=df.corr()
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
print(correlation_matrix)


#calssification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_preds)
print(report)

