# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:11:18 2024

@author: balav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data= pd.read_csv("C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Data Set\\Social_Network_Ads.csv")
x=data.iloc[:,[2,3]].values
y=data.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LogisticRegression
lc=LogisticRegression()
lc.fit(x_train,y_train)

y_pred= lc.predict(x_test)

lc.score(x_train,y_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)

#visualization

from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
x1,x2= np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.001),
                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.001))
plt.contourf(x1,x2,lc.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.5,cmap=ListedColormap('red','green'))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],
                x_set[y_set==j,1],
                c=ListedColormap(('blue'))(i))
plt.show()