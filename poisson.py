# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:36:01 2024

@author: balav
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
x=np.arange(1,13)
pmf=poisson.pmf(x,6)
plt.vlines(x,0,pmf,colors='red',linestyles='-',lw=8)
plt.ylabel('Probability')
plt.xlabel('Intervals')
plt.show()

#poisson cumulative distribution
 cdfp= poisson.cdf(17,mu=12)
 prob=1-cdfp
 
 #range
 cdfp1=poisson.cdf(10,mu=12)
 cdfp2=poisson.cdf(15,mu=12)
 cdf= cdfp2-cdfp1
 
 
 
 #Random variable
data_poisson=poisson.rvs(mu=12,size=10)
ax= sns.distplot(data_poisson,bins=4,kde=True,color='blue',hist_kws={'linewidth':15})
ax.set(xlabel='Binom',ylabel='Frequency')
