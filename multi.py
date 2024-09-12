# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:59:48 2024

@author: balavishwatharan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data= pd.read_csv('C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Data Set\\Social_Network_Ads.csv')
x=data.iloc[:,1:4].values
y=data.iloc[:,-1].values

#Below statement throws error because data has string in it
#sns.heatmap(data.corr(),annot=True,cmap='RdYlBu',center=0) -> returns error due to presence of string
sns.pairplot(data)

#To overcome the error we are converting string into int or float using encoder
#OneHotEncoder adds additional features and returns in float
#LableEncoder converts the string into int

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
x1=x.copy()
x1[:,0]=label.fit_transform(x1[:,0])

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x1=ss.fit_transform(x1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x1,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
regressor.score(x_train,y_train)
regressor.intercept_
regressor.coef_
y_pred=regressor.predict(x_test)
regressor.score(x_test,y_test)
 