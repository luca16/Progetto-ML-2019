# https://www.analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import os
import sys
from define import MINIMUM_ACCURACY


""" GLOBAL VARIABLES """

GAMMA = 1

C_ERROR = 1

INCR_GAMMA = 10
INCR_C_ERROR = 10

KFOLD_SPLITS = 3


""" FUNCTIONS """


def svm_train_and_validate(X_values, Y_values):
    count = 0

    print(" dentro svm train and validate valori xvalues yvalues")
    print(X_values)
    print()
    print(Y_values)
    
    gamma_value = GAMMA
    c_value = C_ERROR
    
    classifier = SVC(kernel='rbf', random_state = 1, gamma=gamma_value, C=c_value)
    
   
    skf = KFold(n_splits=KFOLD_SPLITS, random_state=None, shuffle=False)
    
    for train_idx, validation_idx in skf.split(X_values):
        count += 1
        X_train = []
        Y_train = []
        X_validation = []
        Y_validation = []
        for idx in train_idx:
            X_train.append(X_values[idx])
            Y_train.append(Y_values[idx])
        for idx in validation_idx:
            X_validation.append(X_values[idx])
            Y_validation.append(Y_values[idx])
        print(Y_train)
        classifier.fit(X_train, Y_train)
        accuracy = svm_test(classifier, X_validation, Y_validation)
        
        if accuracy < MINIMUM_ACCURACY:
            gamma_value += INCR_GAMMA
            c_value += INCR_C_ERROR
            classifier = SVC(kernel='rbf', random_state = 1, gamma=gamma_value, C=c_value)
            if count == KFOLD_SPLITS:
                classifier.fit(X_train, Y_train)

    return classifier


def svm_test(classifier, X_values, Y_values):
    Y_pred = classifier.predict(X_values)

    
    cm = confusion_matrix(Y_values,Y_pred)
    accuracy = float(cm.diagonal().sum())/len(Y_values)
    
    return accuracy

