# -*- coding: utf-8 -*-
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
import numpy as np
import sys
sys.path.append('../')
import methods.svm_classifier as svmc


class Svm_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train):
        if (search_HP):
            self.svmcTuning(x_train, y_train)
        else:
            self.svmcDefault()

    def svmcTuning(self, x_train, y_train):
        """
        When searching the best hyperparameters
        """
        params = {'C': np.logspace(-6, 6, 13)}
        print("Start : SVM classifier tuning - research of hyperparameters")
        gd = GridSearchCV(SVC(), params, verbose=3)
        gd.fit(x_train, y_train)
        print("End : SVM classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = svmc.Svm_Classifier(C=gd.best_params_["C"])
    def svmcDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = svmc.Svm_Classifier(C=10) #Best Param in fact


    def getClassifier(self):
        return self.classifier




