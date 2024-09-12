# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:18:11 2024

@author: balav
"""
#####EXAMPLE 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Data Set\\Salary_Data.csv')
x=data.iloc[:,0:1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
regressor.intercept_
regressor.coef_ 
y_pred= 9312.57512673*(9.5)+26780.09915062818
pred_y= regressor.predict(x_test)
regressor.score(x_train,y_train)
regressor.score(x_test,y_test)

#Visulaization
 plt.figure(dpi=300)
 plt.scatter(x_train,y_train,color='red')
 plt.plot(x_train,regressor.predict(x_train),color='blue')
 plt.title('SIMPLE')
 plt.xlabel('Exp')
 plt.ylabel('Salary')
 
 
 
#For checking the efficiency of the model with the problem statement 
from sklearn.metrics import r2_score
r2_score(y_test,pred_y)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,pred_y)

np.sqrt(mean_squared_error(y_test,pred_y)) 

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,pred_y)



#calculating variance and bias

from mlxtend.evaluate import bias_variance_decomp
mse,bias,variance= bias_variance_decomp(regressor,x_train,y_train,x_test,y_test,loss='mse',num_rounds=5,random_seed=123)
