# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 19:47:01 2018

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset1 = pd.read_csv('test.csv')
dataset2 = pd.read_csv('sample_submission.csv')
X = dataset1.iloc[:, :].values
y = dataset2.iloc[:,1].values