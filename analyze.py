#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from define import *
import numpy as np

""" GLOBAL VARIABLES """

# training data
HEALT_MODEL_FILE = 'model/{}_{}_healt_svm.pkl'
SLEEP_MODEL_FILE = 'model/{}_{}_sleep_svm.pkl'
ACTIVITY_MODEL_FILE = 'model/{}_{}_activity_svm.pkl'

DATA_FILE = 'data_analisys.csv'

df1 = pd.read_csv(DATA_FILE, names=['data_type','value'])

age = 0
height = 0
sex = ""

healt_classifier = SVC()
sleep_classifier = SVC()
activity_classifier = SVC()

healt_values = []
sleep_values = []
activity_values = []
value = []

""" FUNCTIONS """

def read_input_data():
    print("si assume che l'utente inserisca numeri e valori validi")
    global age 
    age = input("Please enter your age: ")
    global sex 
    sex = input("Please enter your sex (M for Male and F for Female): ")
    global height 
    height = input("Please enter your height: ")

    

    
def read_training_data():
    for max_age in MAX_AGE_GROUPS:
        if (int(age) <= max_age):
            healt_model_file = HEALT_MODEL_FILE.format(sex, age)
            sleep_model_file = SLEEP_MODEL_FILE.format(sex, age)
            activity_model_file = ACTIVITY_MODEL_FILE.format(sex, age)
            
            with open(healt_model_file, 'rb') as fid:
                global healt_classifier
                healt_classifier = cPickle.load(fid)
            with open(sleep_model_file, 'rb') as fid:
                global sleep_classifier
                sleep_classifier = cPickle.load(fid)
            with open(activity_model_file, 'rb') as fid:
                global activity_classifier
                activity_classifier = cPickle.load(fid)
            
         
def make_groups_values():
    
    for attribute in HEALT_GROUP:
        
        attribute_value = (df1[df1['data_type'] == attribute].value).values[0]
        healt_values.append(attribute_value)
        value.append(attribute_value)
   
        
    for attribute in SLEEP_GROUP:
        attribute_value = (df1[df1['data_type'] == attribute].value).values[0] 
        sleep_values.append(attribute_value)
        value.append(attribute_value)
        
    
    for attribute in ACTIVITY_GROUP:
        attribute_value = (df1[df1['data_type'] == attribute].value).values[0] 
        activity_values.append(attribute_value)
        value.append(attribute_value)

def print_predictions(pred):
    print(pred)

def make_predictions():
    print(healt_values)
    print(sleep_values)
    print(activity_values)
    healt_pred = healt_classifier.predict([healt_values])
    sleep_pred = sleep_classifier.predict([sleep_values])
    activity_pred = activity_classifier.predict([activity_values])
    
    print_predictions(healt_pred)
    print_predictions(sleep_pred)
    print_predictions(activity_pred)
    

    
""" MAIN """

read_input_data()
read_training_data()
make_groups_values()
make_predictions()

