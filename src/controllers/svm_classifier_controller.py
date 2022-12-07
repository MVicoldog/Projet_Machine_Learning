# -*- coding: utf-8 -*-
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
import numpy as np
import sys
sys.path.append('../')
import methods.svm_classifier as svmc
import visualizers.svm_visualizer as svmv


class Svm_Classifier_Controller:

    def __init__(self, search_HP, x_train, y_train):
        if (search_HP):
            self.svmcTuning(x_train, y_train)
        else:
            self.svmcDefault()

    def svmcTuning(self, x_train, y_train, bCv=True):
        """
        When searching the best hyperparameters
        """
        intervale = np.logspace(-6, 6, 13)
        params = {'C': intervale}
        print("Start : SVM classifier tuning - research of hyperparameters")
        gd = GridSearchCV(SVC(), params, verbose=3)
        if bCv:
            gd = GridSearchCV(estimator=SVC(), 
                    param_grid=params, 
                    cv = 5, #Stratified k-fold
                    verbose=2, 
                    scoring='accuracy') 
        else:
            gd = GridSearchCV(estimator=SVC(), 
                    param_grid=params, 
                    verbose=2, 
                    scoring='accuracy')  
        gd.fit(x_train, y_train)
        print("End : SVM classifier tuning - research of hyperparameters")
        model = gd.best_estimator_
        print(gd.best_params_)
        print(gd.best_score_)

        self.classifier = svmc.Svm_Classifier(C=gd.best_params_["C"])
        self.visualizer = svmv.svm_visualizer(gd, intervale)

    def svmcDefault(self):
        """
        When taking default hyperparameters
        """
        self.classifier = svmc.Svm_Classifier(C=10) #Best Param in fact


    def getClassifier(self):
        return self.classifier

    def getVisualizer(self):
        return self.visualizer



