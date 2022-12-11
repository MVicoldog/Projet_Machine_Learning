import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import methods.adaBoost_classifier as adac


class adaBoost_visualizer:

    def __init__(self, model, learning_rate_list, n_estimators_list, x_train, y_train, x_test, y_test, grid):
        self.model = model
        # self.learning_rate_list = learning_rate_list
        # self.n_estimators_list = n_estimators_list
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.grid = grid

        self.best_param_n = grid.best_params_["n_estimators"]
        self.best_param_lr = grid.best_params_["learning_rate"]

        self.intervale_n = [i for i in range(200, 300, 5)]
        self.learning_rate_list = np.arange(1, 4.5, 0.5)

    def Visualise_tuning(self):

        accuracy_over_n = []
        accuracy_over_lr = []
        model = adac.AdaBoostClassifier()

        for n in self.intervale_n:
            model.set_params(n_estimators=n, learning_rate=self.best_param_lr)
            model.fit(self.x_train, self.y_train)
            accuracy_over_n.append(model.score(self.x_test, self.y_test))
        for lr in self.learning_rate_list:
            model.set_params(n_estimators=self.best_param_n, learning_rate=lr)
            model.fit(self.x_train, self.y_train)
            accuracy_over_lr.append(model.score(self.x_test, self.y_test))

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        fig.suptitle('Accuracy over estimators')

        sns.lineplot(ax=axes[0], x=self.intervale_n, y=accuracy_over_n)
        axes[0].set_title('Accuracy over n_estimators')
        axes[0].set_xlabel("n_estimators")
        axes[0].set_ylabel("Accuracy on test data")

        sns.lineplot(ax=axes[1], x=self.learning_rate_list, y=accuracy_over_lr)
        axes[1].set_title('Accuracy over learning_rate')
        axes[1].set_xlabel("learning_rate")
        axes[1].set_ylabel("Accuracy on test data")

        plt.show()
