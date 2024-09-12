# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:26:46 2024

@author: balav
"""
from scipy.stats import norm
import seaborn as sns
import numpy as np
data= np.arange(1,101)
data.mean()
data.std()
##Visualization
pdf=norm.pdf(data,loc=50.5,scale=28.86)
sns.lineplot(pdf,color='red')
cdf=norm.cdf(data,loc=50.5,scale=28.86)
sns.lineplot(cdf,color='red',alpha=0.8)

cdfn=norm(loc=80,scale=15).cdf(80)
prob=1-cdfn
val_60=norm(loc=60,scale=15).cdf(60)
val_80=norm(loc=60,scale=15).cdf(80)
prob=val_80-val_60
###Data normalization
data_nor= norm.rvs(size=100,loc=50.0,scale=28.86)
ax= sns.distplot(data_nor,bins=10,kde=True,color='green',hist_kws={'linewidth':10,'alpha':0.7})
ax.set(xlabel='ND',ylabel='Frequency')
