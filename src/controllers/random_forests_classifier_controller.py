# -*- coding: utf-8 -*-
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier
import numpy as np

sys.path.append('../')
import methods.random_forests_classifier as rf


class Random_Forests_Classifier_Controller:

    def __init__(self, search_HP, x_test, y_train):
        if (search_HP):
            self.rfTuning(x_test, y_train)
        else:
            self.rfDefault()

    def rfTuning(self, x_test, y_train):
        """
        When searching the best hyperparameters
        """
        params = {'n_estimators':[i for i in range(45,48)],  'max_depth': [i for i in range(17,19)]}
        print("Start : random forests tuning - research of hyperparameters")
        gd = GridSearchCV(RandomForestClassifier(), params, verbose=1)
        gd.fit(x_test, y_train)
        print("End : random forests - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = rf.Random_forests_Classifier(n_estimators=gd.best_params_["n_estimators"], max_depth=gd.best_params_["max_depth"])

    def rfDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = rf.RandomForestClassifier(n_estimators = 40, max_depth=15) #Best Param in fact


    def getClassifier(self):
        return self.classifier




