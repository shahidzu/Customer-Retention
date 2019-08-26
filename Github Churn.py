# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:01:45 2019

@author: shahi
"""

#### Importing Libraries ####

import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt

dataset = pd.read_csv('churn_data.csv') # Users who were 60 days enrolled, churn in the next 30

dataset.head() # shows first 5 rows not including column names

dataset.columns #shows column names

dataset.describe() # Distribution of Numerical Variables

dataset.isna().sum() #returns a number, the sume of all items that are N/A's

dataset.isna().any() #returns true for N/A  and false for proper values 

dataset=dataset[pd.notnull(dataset['age'])] #detect non-missing values and updating dataset to only values where age is not null
                                            # meaning left with 26,996 values instead of 27,000 values, loss of 4 values

dataset=dataset[pd.notnull(dataset['credit_score'])] #same as above for credit score
#dataset = dataset.drop(columns = ['credit_score', 'rewards_earned']) #removes mentioned columns (credit score & rewards earned from dataset

dataset["rewards_earned"].fillna(0, inplace=True)# replaces nan to 0

dataset.info() #shows column names, number of non-null values in each column & type of data int(whole)/float(decimal)/ object  etc 


## Correlation Matrix
sn.set(style="white") #seaborn aesthetic style of the plots

# Compute the correlation matrix
corr = dataset.corr()
corr.head()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.2, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

Newdata=pd.read_csv('new_churn_data.csv')


Newdata.head()

user_identifier = Newdata['user']#creates a seprate series for  user

dataset1 = Newdata.drop(columns = ['user'])#dropped first column user

# One-Hot Encoding
dataset1.housing.value_counts()
dataset1.describe()

#import pandas as pd
#cat = pd.Categorical(['age'])
#cat.dtype.name


dataset1 = pd.get_dummies(dataset1)#convert categorical variable into dummy/indicator variables

dataset1.columns

#dataset1 = dataset1.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])
dataset1.info()

# Splitting the dataset into the Training set and Test set, x,y, 20% test data, creates 4 new dataframe/series x_train,x_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset1.drop(columns = 'churn'), dataset1['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)

y_train.value_counts()#showing count of 0/1 in the churn target column

# Balancing the Training Set


pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index): #len function returns number of items/length in an object
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index
lower
random.seed(0)
higher = np.random.choice(higher, size=len(lower))

lower = np.asarray(lower)#to convert input to an array
new_indexes = np.concatenate((lower, higher))#combines lower & higher indexes

X_train = X_train.loc[new_indexes,]#access a group of rows and columns by labels or boolean array
y_train = y_train[new_indexes]

# Feature Scaling/ normalization or standardizng(standardize features by removing the mean and scaling to unit variance)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))#To center the data (make it zero mean and unit standard error you subtract the mean and then a divide the result by thge standard deviation)
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

#### Model Building ####

### Comparing Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Linear Regression (Lasso)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results

## SVM (rbf)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
results
## Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

results

## Confusion Matrix
cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

results

## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))

#### Feature Selection ####


## Feature Selection(improve accuracy or boost performance)
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

X_train.shape

# Select Best X Features
rfe = RFE(classifier, 20)
rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
X_train.columns[rfe.support_]

corr = X_train[X_train.columns[rfe.support_]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

## Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest after new Feature selection', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results.append(model_results, ignore_index = True)
results

## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))


# Formatting Final Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)



