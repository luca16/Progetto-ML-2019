#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import _pickle as cPickle
from define import *
from svm import * 
from kmeans import * 



""" GLOBAL VARIABLES """



DATA_FILE = 'dataset/data.csv'
USERINFO_FILE = 'dataset/userinfo.csv'


DEFINE_FILE = 'define.py'

HEALT_TMP_FILE = 'tmp/{}_{}_healt_svm.pkl'
SLEEP_TMP_FILE = 'tmp/{}_{}_sleep_svm.pkl'
ACTIVITY_TMP_FILE = 'tmp/{}_{}_activity_svm.pkl'

HEALT_MODEL_FILE = 'model/{}_{}_healt_svm.pkl'
SLEEP_MODEL_FILE = 'model/{}_{}_sleep_svm.pkl'
ACTIVITY_MODEL_FILE = 'model/{}_{}_activity_svm.pkl'


df1 = pd.read_csv(DATA_FILE, names=['id','user_id','date','data_type','value'])
df1['date'] = pd.to_datetime(df1['date'])


df2 = pd.read_csv(USERINFO_FILE, names=['id','user_id','b','c','sex','age','height'])


MEAN_STD_VALUES = {   1 : {'min_val': 5000,'max_val': 12000},
            3 : {'min_val': 21, 'max_val': 25},
            4 : {'min_val': 100, 'max_val': 120},
            5 : {'min_val': 9, 'max_val': 11},
            6 : {'min_val': 2, 'max_val': 2},
            7 : {'min_val': 70, 'max_val': 80},
            8 : {'min_val': 60 , 'max_val': 70},
            9 : {'min_val': 90, 'max_val': 110},
            10 : {'min_val': 5, 'max_val': 7},
            15 : {'min_val': 1, 'max_val': 2},
            16 : {'min_val': 1, 'max_val': 2},
            18 : {'min_val': 2, 'max_val': 4},
            19 : {'min_val': 2, 'max_val': 4},
            21 : {'min_val': 700, 'max_val': 3600},
            22 : {'min_val': 700, 'max_val': 1000}
        }
## DATI UTENTE

SEX_LIST = ['Male', 'Female']

# gruppi utenti
training_users = []
test_users = []

### valori medi

training_mean_values = []
test_mean_values = []
               
training_healt_evaluations = []
training_sleep_evaluations = []
training_activity_evaluations = []

test_healt_evaluations = []
test_sleep_evaluations = []
test_activity_evaluations = []



""" FUNCTIONS """

def split_train_test_users():
    training_u = []
    test_u = []

    training_u, test_u = train_test_split(df2['user_id'], test_size=USER_TEST_PERC, random_state=42)

    for user in training_u:
        training_users.append(user)

    for user in test_u:
        test_users.append(user)


def create_mean_values(user_list, mean_values_list):
     
   for person in user_list : 
        person_values = {}
        person_values['user_id'] = person
        person_values['age'] = ((df2['age'].loc[df2['user_id'] == person]).values)[0]  
        person_values['sex'] = ((df2['sex'].loc[df2['user_id'] == person]).values)[0]

        person_values['values'] = []
        
      
     
        for i in range(1,13):

             
            month_mean_value = {}
            month_mean_value['month'] = i
            month_data = df1[(df1['date'].dt.month == i) & (df1['user_id']== person)]
           
           
            for attribute in HEALT_GROUP:
                attr_month_data = month_data[month_data['data_type'] == attribute] 
                
                
                if(len(attr_month_data) < MINIMUM_READINGS):
                    month_mean_value[attribute] = None 
                else :
                    attribute_mean_value = round(attr_month_data['value'].mean(),2)
                    month_mean_value[attribute] = attribute_mean_value
                
            for attribute in SLEEP_GROUP:
                attr_month_data = month_data[month_data['data_type'] == attribute] 
                
                          
                if(len(attr_month_data) < MINIMUM_READINGS):
                    month_mean_value[attribute] = None 
                else:
                    attribute_mean_value = round(attr_month_data['value'].mean(),2)
                    month_mean_value[attribute] = attribute_mean_value
            

            for attribute in ACTIVITY_GROUP:
                attr_month_data = month_data[month_data['data_type'] == attribute] 
                
                
                
                if(len(attr_month_data) < MINIMUM_READINGS):
                    month_mean_value[attribute] = None 
                else:
                    attribute_mean_value = round(attr_month_data['value'].mean(),2)
                    month_mean_value[attribute] = attribute_mean_value
            
            person_values['values'].append(month_mean_value)
      
        
        mean_values_list.append(person_values) 


    
def create_evaluation_values(mean_values, healt_evaluations, sleep_evaluations, activity_evaluations):
   
    for person in mean_values:
      
        for month_values in person['values']:
            
          
            healt_evaluation = {}
            healt_evaluation['user_id'] = person['user_id']
            healt_evaluation['age'] = person['age']
            healt_evaluation['sex'] = person['sex']
            healt_evaluation['month'] = month_values['month']
            healt_evaluation['evaluation'] = 0
            
          
            null_attribute = False
            
            for attribute in HEALT_GROUP:
                
               
                attr_min_val = MEAN_STD_VALUES[attribute]['min_val']
                attr_max_val = MEAN_STD_VALUES[attribute]['max_val']
                
                if(month_values[attribute] == None):
                    null_attribute = True
                    break
                
                if((month_values[attribute] >= attr_min_val) & (month_values[attribute] <= attr_max_val)):
                    healt_evaluation['evaluation'] += 1
                else:
                    healt_evaluation['evaluation'] -= 1
            
            
            if(null_attribute == False):
                healt_evaluations.append(healt_evaluation)
            
            
            # sleep_group
            sleep_evaluation = {}
            sleep_evaluation['user_id'] = person['user_id']
            sleep_evaluation['age'] = person['age']
            sleep_evaluation['sex'] = person['sex']
            sleep_evaluation['month'] = month_values['month']
            sleep_evaluation['evaluation'] = 0
            
           
            null_attribute = False
            
            for attribute in SLEEP_GROUP:
                
                attr_min_val = MEAN_STD_VALUES[attribute]['min_val']
                attr_max_val = MEAN_STD_VALUES[attribute]['max_val']
                
                if(month_values[attribute] == None):
                    null_attribute = True
                    break
                
                if((month_values[attribute] >= attr_min_val) & (month_values[attribute] <= attr_max_val)):
                    sleep_evaluation['evaluation'] += 1
                else:
                    sleep_evaluation['evaluation'] -= 1
            
           
            if(null_attribute == False):
                sleep_evaluations.append(sleep_evaluation)
            
            
            # activity_group
            activity_evaluation = {}
            activity_evaluation['user_id'] = person['user_id']
            activity_evaluation['age'] = person['age']
            activity_evaluation['sex'] = person['sex']
            activity_evaluation['month'] = month_values['month']
            activity_evaluation['evaluation'] = 0
            
          
            null_attribute = False
            
            for attribute in ACTIVITY_GROUP:
                
                
                attr_min_val = MEAN_STD_VALUES[attribute]['min_val']
                attr_max_val = MEAN_STD_VALUES[attribute]['max_val']
                
                if(month_values[attribute] == None):
                    null_attribute = True
                    break
                
                if((month_values[attribute] >= attr_min_val) & (month_values[attribute] <= attr_max_val)):
                    activity_evaluation['evaluation'] += 1
                else:
                    activity_evaluation['evaluation'] -= 1
            
            
            if(null_attribute == False):
                activity_evaluations.append(activity_evaluation)
            

def assign_class_to_evaluations(healt_evaluation, sleep_evaluation, activity_evaluation):
    
    # healt kmeans
    healt_kmeans = []
    for evaluation in healt_evaluation:
        healt_kmeans.append([evaluation['evaluation']])

    # sleep kmeans
    sleep_kmeans = []
    for evaluation in sleep_evaluation:
        sleep_kmeans.append([evaluation['evaluation']])

    # activity kmeans
    activity_kmeans = []
    for evaluation in activity_evaluation:
        activity_kmeans.append([evaluation['evaluation']])

            
    # calcola le classi
    print(healt_kmeans)
    healt_assignment = k_means(healt_kmeans, CLASS_NUMBER)
    sleep_assignment = k_means(sleep_kmeans, CLASS_NUMBER)
    activity_assignment = k_means(activity_kmeans, CLASS_NUMBER)


      
    for i in range(len(healt_evaluation)):
        healt_evaluation[i]['class'] = healt_assignment[i]
        
    for i in range(len(sleep_evaluation)):
        sleep_evaluation[i]['class'] = sleep_assignment[i]

    for i in range(len(activity_evaluation)):
        activity_evaluation[i]['class'] = activity_assignment[i]


def classificator_training():
    for sex in SEX_LIST:
        for max_age in MAX_AGE_GROUPS:
            svm_training(sex, max_age)


def classificator_testing():
    for sex in SEX_LIST:
        for max_age in MAX_AGE_GROUPS:
            svm_testing(sex, max_age)

def getMeanValues(dataset,user_id, month):
    mean_value = []
    for user in dataset:
        if ((user['user_id']== user_id)):
            values = user['values']
            for value in values:
                if (value['month'] == month):
                    mean_value = value
    return mean_value

def svm_training(sex, max_age):
    # healt_svm
    healt_values = []
    healt_classes = []

    for evaluation in training_healt_evaluations:
       
        if (evaluation['sex'] == sex):
            
            if (evaluation['age'] <= max_age):
                array_values = []
                user_values = getMeanValues(training_mean_values,(evaluation['user_id']),(evaluation['month']))
                for attribute in HEALT_GROUP:
                    array_values.append(user_values[attribute])
                healt_values.append(array_values)
                healt_classes.append(evaluation['class'])
   
    classificator = svm_train_and_validate(healt_values, healt_classes)
    dump_classificator(classificator, sex, max_age, HEALT_TMP_FILE)



    # sleep_svm
    sleep_values = []
    sleep_classes = []
    for evaluation in training_sleep_evaluations:
        if evaluation['sex'] == sex:
            if evaluation['age'] <= max_age:
                array_values = []
                user_values = getMeanValues(training_mean_values, evaluation['user_id'], evaluation['month'])
                for attribute in SLEEP_GROUP:
                    array_values.append(user_values[attribute])
                sleep_values.append(array_values)
                sleep_classes.append(evaluation['class'])
        
    classificator = svm_train_and_validate(sleep_values, sleep_classes)
    dump_classificator(classificator, sex, max_age, SLEEP_TMP_FILE)
    
    # activity_svm
    activity_values = []
    activity_classes = []
    for evaluation in training_activity_evaluations:
        if evaluation['sex'] == sex:
            if evaluation['age'] <= max_age:
                array_values = []
                user_values = getMeanValues(training_mean_values, evaluation['user_id'], evaluation['month'])
                for attribute in ACTIVITY_GROUP:
                    array_values.append(user_values[attribute])
                activity_values.append(array_values)
                activity_classes.append(evaluation['class'])
        
    classificator = svm_train_and_validate(activity_values, activity_classes)
    dump_classificator(classificator, sex, max_age, ACTIVITY_TMP_FILE)
    
    
def svm_testing(sex, max_age):
    # healt_svm
    healt_values = []
    healt_classes = []
    for evaluation in test_healt_evaluations:
        array_values = []
        user_values = getMeanValues(test_mean_values, evaluation['user_id'], evaluation['month'])
        for attribute in HEALT_GROUP:
            array_values.append(user_values[attribute])
        healt_values.append(array_values)
        healt_classes.append(evaluation['class'])
        
    # sleep_svm
    sleep_values = []
    sleep_classes = []
    for evaluation in test_sleep_evaluations:
        array_values = []
        user_values = getMeanValues(test_mean_values, evaluation['user_id'], evaluation['month'])
        for attribute in SLEEP_GROUP:
            array_values.append(user_values[attribute])
        sleep_values.append(array_values)
        sleep_classes.append(evaluation['class'])
        
    # activity_svm
    activity_values = []
    activity_classes = []
    for evaluation in test_activity_evaluations:
        array_values = []
        user_values = getMeanValues(test_mean_values, evaluation['user_id'], evaluation['month'])
        for attribute in ACTIVITY_GROUP:
            array_values.append(user_values[attribute])
        activity_values.append(array_values)
        activity_classes.append(evaluation['class'])    
        
        
    # calculate accuracy
    classificator = load_classificator(sex, max_age, HEALT_TMP_FILE)
    healt_accuracy = svm_test(classificator, healt_values, healt_classes)
    
    classificator = load_classificator(sex, max_age, SLEEP_TMP_FILE)
    sleep_accuracy = svm_test(classificator, sleep_values, sleep_classes)
    
    classificator = load_classificator(sex, max_age, ACTIVITY_TMP_FILE)
    activity_accuracy = svm_test(classificator, activity_values, activity_classes)
    
    
    if((healt_accuracy < MINIMUM_ACCURACY) | (sleep_accuracy < MINIMUM_ACCURACY) | (activity_accuracy < MINIMUM_ACCURACY)):
        print("ERROR! Accuracy for {} with maximum {} years is".format(sex, max_age))
        print("HEALT ACCURACY = {}".format(healt_accuracy))
        print("SLEEP ACCURACY = {}".format(sleep_accuracy))
        print("ACTIVITY ACCURACY = {}".format(activity_accuracy))
        print("MINIMUM ACCURACY REQUIRED = {}".format(MINIMUM_ACCURACY))
        print("Try to change some parameters in {}".format(DEFINE_FILE))
        #delete_tmp_classifiers(sex, max_age)
    else:
        print("DONE! Accuracy for {} with maximum {} years is".format(sex, max_age))
        print("HEALT ACCURACY = {}".format(healt_accuracy))
        print("SLEEP ACCURACY = {}".format(sleep_accuracy))
        print("ACTIVITY ACCURACY = {}".format(activity_accuracy))
        print("MINIMUM ACCURACY REQUIRED = {}".format(MINIMUM_ACCURACY))
        move_classifiers(sex, max_age)


def load_classificator(sex, max_age, filename):
    if sex == 'Male':
        sex_print = 'M'
    else:
        sex_print = 'F'
        
    age_print = max_age
    if max_age == MAX_AGE:
        age_print = 'MAX'
        
    loadfile = filename.format(sex_print, age_print)
    
    with open(loadfile, 'rb') as fid:
        classifier = cPickle.load(fid)

    return classifier


def dump_classificator(classificator, sex, max_age, filename):
    
    if sex == 'Male':
        sex_print = 'M'
    else:
        sex_print = 'F'
        
    age_print = max_age
    if max_age == MAX_AGE:
        age_print = 'MAX'
        
    savefile = filename.format(sex_print, age_print)
    
    with open(savefile, 'wb') as fid:
        cPickle.dump(classificator, fid) 
        

def delete_tmp_classifiers(sex, max_age):
    
    if sex == 'Male':
        sex_print = 'M'
    else:
        sex_print = 'F'
        
    age_print = max_age
    if max_age == MAX_AGE:
        age_print = 'MAX'
    
    healt_tmp_file = HEALT_TMP_FILE.format(sex_print, age_print)
    sleep_tmp_file = SLEEP_TMP_FILE.format(sex_print, age_print)
    activity_tmp_file = ACTIVITY_TMP_FILE.format(sex_print, age_print)
    
    os.remove(healt_tmp_file)
    os.remove(sleep_tmp_file)
    os.remove(activity_tmp_file)
    
    
def move_classifiers(sex, max_age):
        
    if sex == 'Male':
        sex_print = 'M'
    else:
        sex_print = 'F'
        
    age_print = max_age
    if max_age == MAX_AGE:
        age_print = 'MAX'
    
    healt_tmp_file = HEALT_TMP_FILE.format(sex_print, age_print)
    sleep_tmp_file = SLEEP_TMP_FILE.format(sex_print, age_print)
    activity_tmp_file = ACTIVITY_TMP_FILE.format(sex_print, age_print)
    
    healt_model_file = HEALT_MODEL_FILE.format(sex_print, age_print)
    sleep_model_file = SLEEP_MODEL_FILE.format(sex_print, age_print)
    activity_model_file = ACTIVITY_MODEL_FILE.format(sex_print, age_print)
    
    os.rename(healt_tmp_file, healt_model_file)
    os.rename(sleep_tmp_file, sleep_model_file)
    os.rename(activity_tmp_file, activity_model_file)
    

def create_tmp_folder():
    newpath = 'tmp' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    
""" MAIN """

split_train_test_users()
print("start")

print(len(training_users))
print(len(test_users))

print("creazione set completata")

create_mean_values(training_users, training_mean_values)
create_mean_values(test_users, test_mean_values)

create_evaluation_values(training_mean_values, training_healt_evaluations, training_sleep_evaluations, training_activity_evaluations)
create_evaluation_values(test_mean_values, test_healt_evaluations, test_sleep_evaluations, test_activity_evaluations)

assign_class_to_evaluations(training_healt_evaluations, training_sleep_evaluations, training_activity_evaluations)
assign_class_to_evaluations(test_healt_evaluations, test_sleep_evaluations, test_activity_evaluations)

create_tmp_folder()

classificator_training()
classificator_testing()
