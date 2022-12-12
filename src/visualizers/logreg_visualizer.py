
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import methods.LogReg_classifier as llc

class logreg_visualizer:

    def __init__(self, model, grid, intervale, x_train, y_train, x_test, y_test):
        self.model = model
        # self.learning_rate_list = learning_rate_list
        # self.n_estimators_list = n_estimators_list
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.grid = grid
        self.intervale = intervale

    def Visualise_tuning(self):
        """
        plot model's accuracy depending on each hyper-parameter
        """
        CVresults = self.grid.cv_results_

        ymax = np.ones(len(CVresults["mean_test_score"])) * \
            max(CVresults["mean_test_score"])
        plt.figure(figsize=(5, 5))
        plt.plot(self.intervale, ymax, label='Best value is : ' + '{:1.3f}'.format(max(
            CVresults["mean_test_score"])) + ' for C = ' + '{:1.5f}'.format(list(self.grid.best_params_.values())[0]))

        plt.plot(self.intervale, CVresults["mean_test_score"])
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Value of C')
        plt.ylabel('Accuracy')
        plt.show()

    def  Visualise_penalty(self):
        accuracy_over_L1 = []
        accuracy_over_L2 = []


        model = llc.LogisticRegression(penalty='l1', solver='liblinear')
        for c in self.intervale:
            model.set_params(C = c )
            model.fit(self.x_train, self.y_train)
            accuracy_over_L1.append(model.score(self.x_test, self.y_test))

        model = llc.LogisticRegression(penalty='l2', solver='liblinear')
        for c in self.intervale:
            model.set_params(C = c)
            model.fit(self.x_train, self.y_train)
            accuracy_over_L2.append(model.score(self.x_test, self.y_test))

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        fig.suptitle('Accuracy over penalties l1 and l2')

        sns.lineplot(ax=axes[0], x=self.intervale, y=accuracy_over_L1)
        axes[0].set_title('Accuracy over L1')
        axes[0].set_xlabel("C")
        axes[0].set_ylabel("Accuracy on test data ")
        axes[0].set_xscale('log')

        sns.lineplot(ax=axes[1], x=self.intervale, y=accuracy_over_L2)
        axes[1].set_title('Accuracy over L2')
        axes[1].set_xlabel("C")
        axes[1].set_ylabel("Accuracy on test data")
        axes[1].set_xscale('log')

        plt.show()
