# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:47:01 2018

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset1 = pd.read_csv('test.csv')
dataset2 = pd.read_csv('sample_submission.csv')
X = dataset1.iloc[:, :].values
y = dataset2.iloc[:,1].values

dataset1.head()

sns.set(style="darkgrid")
lot_area = dataset1['LotArea']

sns.distplot(lot_area)
plt.show()

sns.relplot(x='SaleCondition', y=y, data=dataset1)
plt.show()

sns.relplot(x='LotArea', y=y, data=dataset1)
plt.show()

sns.countplot(x='YrSold', data=dataset1)
plt.show()
