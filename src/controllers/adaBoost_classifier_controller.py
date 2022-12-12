# -*- coding: utf-8 -*-
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import sys
sys.path.append('../')
import visualizers.adaBoost_visualizer as abv
import methods.adaBoost_classifier as abc

class adaBoost_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train, x_test, y_test):
        if (search_HP):
            self.abcTuning(x_train, y_train, x_test, y_test)
        else:
            self.abcDefault()

    def abcTuning(self, x_train, y_train, x_test, y_test, bCv=True):
        """
        When searching the best hyperparameters
        """
        n_estimators_list = np.arange(1, 300, 50)

        learning_rate_list = np.arange(1, 5, 0.5)
        params = {'n_estimators': n_estimators_list,
                  'learning_rate': learning_rate_list}
        print("Start : adaBoost classifier tuning - research of hyperparameters")
        gd = GridSearchCV(AdaBoostClassifier(), params, verbose=3)
        if bCv:
            gd = GridSearchCV(estimator=AdaBoostClassifier(),
                              param_grid=params,
                              cv=5,  # Stratified k-fold
                              verbose=2,
                              scoring='accuracy')
        else:
            gd = GridSearchCV(estimator=AdaBoostClassifier(),
                              param_grid=params,
                              verbose=2,
                              scoring='accuracy')
        gd.fit(x_train, y_train)
        print("End : adaBoost classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(model)
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = abc.adaBoost_Classifier(base_estimator=None, n_estimators=gd.best_params_[
                                                  "n_estimators"], learning_rate=gd.best_params_["learning_rate"])  # base_estimator=gd.best_params_["base_estimator"],
        self.visualizer = abv.adaBoost_visualizer(AdaBoostClassifier, learning_rate_list, n_estimators_list,
                                                  x_train, y_train, x_test, y_test, gd)

    def abcDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = abc.adaBoost_Classifier(
            base_estimator=None, n_estimators=50, learning_rate=1)

    def getClassifier(self):
        return self.classifier

    def getVisualizer(self):
        return self.visualizer
