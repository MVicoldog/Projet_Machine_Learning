# -*- coding: utf-8 -*-
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier
import numpy as np

sys.path.append('../')
import methods.random_forests_classifier as rf
import visualizers.random_forest_visualizer as rfv


class Random_Forests_Classifier_Controller:

    

    def __init__(self, search_HP, x_test, y_train):

       

        if (search_HP):
            self.rfTuning(x_test, y_train)
        else: 
            self.rfDefault()


    def rfTuning(self, x_test, y_train, bCv=True):
        """
        When searching the best hyperparameters
        """
        interval_n_estimators = [i for i in range(45,48)]
        interval_max_depth = [i for i in range(17,19)]

        params = {'n_estimators':interval_n_estimators,  'max_depth': interval_max_depth}
        print("Start : random forests tuning - research of hyperparameters")
        if bCv:
            gd = GridSearchCV(estimator=RandomForestClassifier(), 
                    param_grid=params, 
                    cv = 5, #Stratified k-fold
                    verbose=2, 
                    scoring='accuracy') 
        else:
            gd = GridSearchCV(estimator=RandomForestClassifier(), 
                    param_grid=params, 
                    verbose=2, 
                    scoring='accuracy')  
        gd.fit(x_test, y_train)
        print("End : random forests - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = rf.Random_forests_Classifier(n_estimators=gd.best_params_["n_estimators"], max_depth=gd.best_params_["max_depth"])
        self.visualizer = rfv.random_forest_visualizer(gd, interval_max_depth, interval_n_estimators)

    def rfDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = rf.Random_forests_Classifier(n_estimators = 100, max_depth=None)


    def getClassifier(self):
        return self.classifier

    def getVisualizer(self):
        return self.visualizer



