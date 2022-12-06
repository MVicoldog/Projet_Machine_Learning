# -*- coding: utf-8 -*-
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

sys.path.append('../')

import methods.gaussianNB_classifier as gNBc

class gaussianNB_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train):
        if (search_HP):
            self.nbTuning(x_train, y_train)
        else:
            self.nbDefault()

    def nbTuning(self, x_train, y_train, bCv = True):
        """
        When searching the best hyperparameters
        """
        params = {'var_smoothing': np.logspace(0,-9, num=100)}
        print("Start : Gaussian NB tuning - research of hyperparameters")
        if bCv:
            gd = GridSearchCV(estimator=GaussianNB(), 
                    param_grid=params, 
                    cv = 5, #Stratified k-fold
                    verbose=1, 
                    scoring='accuracy') 
        else:
            gd = GridSearchCV(estimator=GaussianNB(), 
                    param_grid=params, 
                    verbose=1, 
                    scoring='accuracy')  
        gd.fit(x_train, y_train)
        print("End : Gaussian NB classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = gNBc.gaussianNB_Classifier(var_smoothing=gd.best_params_["var_smoothing"])

    def nbDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = gNBc.gaussianNB_Classifier(var_smoothing = 1e-9) #Default Param


    def getClassifier(self):
        return self.classifier




