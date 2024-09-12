# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:18:44 2024

@author: balav
"""

import seaborn as sns     # provides relationship between variables
import matplotlib.pyplot as plt   
import numpy as np
from scipy.stats import binom

pb=binom(n=12,p=0.5)   #it requires no of trials and success
x=np.arange(1,13)     #0 to 12 
pmf=pb.pmf(x)    #Probability mass function - to find the probability of discrete variables
plt.figure(dpi=300)   # for clarity
plt.vlines(x,0,pmf,color='red',linestyle='-',lw=5)  #for graph vertical lines
plt.xlabel('Intervals')
plt.ylabel('Probability')
plt.show()
