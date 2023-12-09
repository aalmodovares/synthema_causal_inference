import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
#import predictors of sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#import metrics of sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
#import cross validation
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from preprocess_data import load_data

def print_metrics(target_test, target_pred):
    print(f'accuracy: {accuracy_score(target_test, target_pred)}')
    print(f'precision: {precision_score(target_test, target_pred)}')
    print(f'recall: {recall_score(target_test, target_pred)}')
    print(f'f1: {f1_score(target_test, target_pred)}')
    print(f'roc_auc: {roc_auc_score(target_test, target_pred)}')
    print(f'confusion matrix: \n{confusion_matrix(target_test, target_pred)}')
    print('')

def train_and_test_model(model, predictors_train, target_train, predictors_test, target_test):
    #train the model
    model.fit(predictors_train, target_train)
    #predict the test set
    target_pred = model.predict(predictors_test)
    #compute the metrics
    print_metrics(target_test, target_pred)

#load the data
database, info_dict = load_data()

#try to predict treatments 'Allogenic HSCT' and 'Autologous HSCT' (all binary variables) from the rest of the variables

#split the database in predictors and targets
predictors = database.drop(info_dict['treatments'], axis=1)
targets = database.loc[:, info_dict['treatments']]
#convert to numpy
predictors = predictors.values.astype(float)
targets = targets.values.astype(int)

models = [LogisticRegression(), RandomForestClassifier(), SVC(), MLPClassifier(), KNeighborsClassifier(), GaussianNB()]
model_names = ['Logistic Regression', 'Random Forest', 'SVC', 'MLP', 'KNN', 'Gaussian NB']

for i, treatment in enumerate(info_dict['treatments']):
    print(f'\n\n#### {treatment} ####')
    target = targets[:, i]

    #keep only the indices where target is not null
    _predictors = predictors[~np.isnan(target)]
    _target = target[~np.isnan(target)]

    #impute 0 in _predictors where there is a nan
    _predictors = np.nan_to_num(_predictors)


    #check if target is balanced
    print(f'number of 0: {len(target[target == 0])}')
    print(f'number of 1: {len(target[target == 1])}')
    print(f'percentage of 1: {len(target[target == 1]) / len(target)}')

    #split the data in train and test
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(_predictors, _target):
        predictors_train, predictors_test = _predictors[train_index], _predictors[test_index]
        target_train, target_test = _target[train_index], _target[test_index]

    #train the model
    for model, name in zip(models, model_names):
        print(f'\n-- {name}--')
        train_and_test_model(model, predictors_train, target_train, predictors_test, target_test)






