# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def open_data(train_path='./data/ClassificationFeuilleArbre/train.csv', test_path='./data/ClassificationFeuilleArbre/test.csv'):
    # Load data from path
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def showdown_df():
    # Create blank dataframe to compare classifier accuracy & log loss
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    return log

def showdownPutter(name, acc, ll):
    # Create complete dataframe to compare classifier accuracy & log loss
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    return log_entry

def display_scores(scores):
    # Create dataframe with mean & std variation from argument "scores" list
    score_df = pd.DataFrame({'Scores': scores})
    score_df['Mean'] = score_df['Scores'].mean()
    score_df['Std Variation'] = score_df['Scores'].std()
    return score_df

class GestionDonnees:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, labels=None, classes=None, train_df=None, test_df=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels
        self.classes = classes
        self.train_df, self.test_df = open_data()

    def prepocess(self):
        # Remove useless information in dataframe + numerize labels.
        coder = LabelEncoder().fit(self.train_df.species)
        self.labels = coder.transform(self.train_df.species)
        self.classes = list(coder.classes_)
        self.train_df = self.train_df.drop(['id', 'species'], axis=1)
        self.test_df = self.test_df.drop(['id'], axis=1)

    def stratifiedSelection(self):
        # Compute stratified selection to get a split between training and validating data
        split = StratifiedShuffleSplit(
            n_splits=10, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(self.train_df, self.labels):
            self.x_train, self.x_test = self.train_df.values[
                train_index], self.train_df.values[test_index]
            self.y_train, self.y_test = self.labels[train_index], self.labels[test_index]

        self.normalize()

    def normalize(self):
        # Normalize data
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)
