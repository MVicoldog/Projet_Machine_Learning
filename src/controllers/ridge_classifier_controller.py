# -*- coding: utf-8 -*-
import sys

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit

from sklearn.linear_model import RidgeClassifier
import numpy as np

sys.path.append('../')
import methods.ridge_classifier as rc


class Ridge_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train):
        if (search_HP):
            self.rcTuning(x_train, y_train)
        else:
            self.rcDefault()

    def rcTuning(self, x_train, y_train):
        """
        When searching the best hyperparameters
        """
        params = {'alpha': np.logspace(-6, 6, 13)}
        print("Start : ridge classifier tuning - research of hyperparameters")
        gd = GridSearchCV(RidgeClassifier(), params, verbose=3)
        gd.fit(x_train, y_train)
        print("End : ridge classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = rc.Ridge_Classifier(alpha=gd.best_params_["alpha"])

    def rcDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = rc.Ridge_Classifier(alpha=100) #Best Param in fact


    def getClassifier(self):
        return self.classifier




