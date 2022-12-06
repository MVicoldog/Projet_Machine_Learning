# -*- coding: utf-8 -*-
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

sys.path.append('../')

import methods.LogReg_classifier as lrc

class LogReg_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train):
        if (search_HP):
            self.lgTuning(x_train, y_train)
        else:
            self.lgDefault()

    def lgTuning(self, x_train, y_train, bCv = True):
        """
        When searching the best hyperparameters
        """
        params = {'C':[1, 10, 50, 100, 500, 1000, 2000],  'tol': [0.001,0.005, 0.0001]}
        print("Start : Logistic Regression tuning - research of hyperparameters")
        if bCv:
            gd = GridSearchCV(estimator=LogisticRegression(), 
                    param_grid=params, 
                    cv = 5, #Stratified k-fold
                    verbose=1, 
                    scoring='accuracy') 
        else:
            gd = GridSearchCV(estimator=LogisticRegression(), 
                    param_grid=params, 
                    verbose=1, 
                    scoring='accuracy')  
        gd.fit(x_train, y_train)
        print("End : Logistic Regression classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = lrc.LogReg_Classifier(C=gd.best_params_["C"], tol=gd.best_params_["tol"])

    def lgDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = LogisticRegression(C=1, tol=1e-4) #Default Param


    def getClassifier(self):
        return self.classifier




