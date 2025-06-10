#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 20:54:59 2025

@author: uhadmin
"""

import pickle

import numpy as np

local_classifier = pickle.load(open('classifier.picle','rb'))
local_scaler = pickle.load(open('sc.pickle','rb'))

new_pred = local_classifier.predict(local_scaler.transform(np.array([[40,20000]])))

new_pred_prob = local_classifier.predict_proba(local_scaler.transform(np.array([[40,20000]])))[:,1]

new_pred2 = local_classifier.predict(local_scaler.transform(np.array([[42,50000]])))

new_pred_prob2 = local_classifier.predict_proba(local_scaler.transform(np.array([[42,50000]])))[:,1]