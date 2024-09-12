# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:06:26 2024

@author: balav
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Data Set\\Social_Network_Ads.csv') 
x=data.iloc[:,2:4].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.svm import SVR
regressor= SVR(kernel='poly')
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
