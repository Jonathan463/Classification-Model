#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:02:05 2025

@author: uhadmin
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv('cleaned_purchase_data.csv')

training_data.describe()

X = training_data.iloc[:,:-1].values
y = training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""## Build a Classification model 
### I am using KNN Classifier in this example
metric = "minkowski"
"""

from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance 
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

##Model training
classifier.fit(X_train, y_train)