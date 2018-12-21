# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:47:01 2018

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, dataset.columns != 'SalePrice'].values
y = dataset['SalePrice'].values

sns.set(style="darkgrid")
lot_area = dataset['LotArea']

sns.distplot(lot_area)
plt.show()

sns.relplot(x='SaleCondition', y=y, data=dataset)
plt.show()

sns.relplot(x='LotArea', y=y, data=dataset)
plt.show()

sns.countplot(x='YrSold', data=dataset)
plt.show()

sns.heatmap(dataset.corr(), xticklabels=1, yticklabels=1)
plt.show()
