# -*- coding: utf-8 -*-
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np

sys.path.append('../')

import methods.neural_network_classifier as nnc

class Neural_Network_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train):
        if (search_HP):
            self.nnTuning(x_train, y_train)
        else:
            self.nnDefault()

    def nnTuning(self, x_train, y_train, bCv = True):
        """
        When searching the best hyperparameters
        """
        intervale = [i for i in range(10,30)]
        params = {'hidden_layer_sizes': intervale}
        print("Start : Neural Network tuning - research of hyperparameters")
        if bCv:
            gd = GridSearchCV(estimator=MLPClassifier(), 
                    param_grid=params, 
                    cv = 5, #Stratified k-fold
                    verbose=1, 
                    scoring='accuracy') 
        else:
            gd = GridSearchCV(estimator=MLPClassifier(), 
                    param_grid=params, 
                    verbose=1, 
                    scoring='accuracy')  

        gd.fit(x_train, y_train)
        print("End : Neural Network classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = nnc.Neural_Network_Classifier(gd.best_params_['hidden_layer_sizes'])

        self.visualizer = nnc.Neural_Network_Classifier(gd, intervale)

    def nnDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = MLPClassifier(hidden_layer_sizes=(28,)) #Default Param


    def getClassifier(self):
        return self.classifier

    def getVisualizer(self):
        return self.visualizer



