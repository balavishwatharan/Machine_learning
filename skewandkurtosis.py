# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:45:33 2024

@author: balav
"""

#skewness and kurtosis
#skewness= refers to degree of symmetry
#kurtosis = refers to degree of outliers(flatness)

import pandas as pd
pd=pd.read_csv('C:\\Users\\balav\\OneDrive\\Desktop\\Imagecon\\Data Set\\mba.csv')
pd.skew() 
pd.kurt()
