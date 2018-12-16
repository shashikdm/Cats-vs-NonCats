#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:01:47 2018

@author: shashi
"""

import numpy as np
from DNN import DNN
from dnn_app_utils_v3 import *
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
dnn = DNN(2,[7,1],train_x,train_y,learning_rate = 0.0075, iterations = 2500)
dnn.train()
probs, predictions = dnn.predict(train_x)

accuracy = np.sum(predictions == train_y)/train_y.shape[1]
print("accuracy = ", accuracy)