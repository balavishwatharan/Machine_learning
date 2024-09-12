# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:52:31 2024

@author: balav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Data Set\\Iris_new.csv")
x=data.iloc[:,0:4].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()

from sklearn.naive_bayes import MultinomialNB
classifier= MultinomialNB()

from sklearn.naive_bayes import BernoulliNB
classifier= BernoulliNB()

classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
