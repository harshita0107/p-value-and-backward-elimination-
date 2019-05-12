# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:28:34 2019

@author: user
"""
import numpy as np
import pandas as pd

#import the dataset
dataset=pd.read_csv('D:/breast_cancer.csv', index_col= False)

#remove the blank column and the id column
dataset.drop('Unnamed: 32', axis=1, inplace= True)
dataset.drop('id', axis=1, inplace= True)

#setting the X and y
X= dataset.iloc[:,1:29].values
y=dataset.iloc[:,0].values

#performing the labelencoding 
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y= labelencoder_y.fit_transform(y)

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state= 0)

#standardising the values
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#creating model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predicting thr values on test data set
y_pred= classifier.predict(X_test)

#checking the confusion matrix and accuracy 
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
score= accuracy_score(y_test, y_pred)

#iporting the library and adding the constant column
import statsmodels.formula.api as sm
X= np.append(arr= np.ones((569, 1)).astype(int), values= X, axis=1)

#defining X_opt and fitting to OLS 
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x2 with highest p value
X_opt = X[:, [0,1,3, 4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#removing the x12 variable
X_opt = X[:, [0,1,3, 4,5,6,7,8,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove the x5
X_opt = X[:, [0,1,3,4,6,7,8,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove the x19
X_opt = X[:, [0,1,3, 4,6,7,8,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove the x16
X_opt = X[:, [0,1,3, 4,6,7,8,10,11,12,13,15,16,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x12
X_opt = X[:, [0,1,3, 4,6,7,8,10,11,12,13,15,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x2
X_opt = X[:, [0,1, 4,6,7,8,10,11,12,13,15,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x8
X_opt = X[:, [0,1, 4,6,7,8,10,11,13,15,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x2
X_opt = X[:, [0,1, 4,7,8,10,11,13,15,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x3
X_opt = X[:, [0,1, 4,7,10,11,13,15,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x4
X_opt = X[:, [0,1, 4,7,10,13,15,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x5
X_opt = X[:, [0,1, 4,7,10,13,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x5
X_opt = X[:, [0,1, 4,7,10,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x4
X_opt = X[:, [0,1, 4,7,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()

#remove x3
X_opt = X[:, [0,1, 4,17,18,19,21,22,24,25,26,27]]
regressor_ols= sm.OLS(endog= y, exog= X_opt).fit()
regressor_ols.summary()


x_train, x_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size=0.2, random_state= 0)
classifier.fit(x_train, y_opt_train)
y_pred_opt = classifier.predict(x_test)
new_score=accuracy_score(y_opt_test, y_pred_opt)
cm_new=confusion_matrix(y_opt_test, y_pred_opt)
         

