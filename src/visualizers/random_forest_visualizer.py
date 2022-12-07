
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import plotly.express as px

import methods.random_forests_classifier as rfc

class random_forest_visualizer:

    def __init__(self, model, intervale_n, intervale_max_d, x_train, y_train, x_test, y_test, grid):
        self.model = model
        # self.intervale_n = intervale_n
        # self.intervale_max_d = intervale_max_d 
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.grid = grid

        self.best_param_n = grid.best_params_["n_estimators"]
        self.best_param_max_d = grid.best_params_["max_depth"]

        self.intervale_n = [i for i in range(1,100)]
        self.intervale_max_d = [i for i in range(1,100)]
 
    
    def Visualise_tuning(self):

        accuracy_over_n =[]
        accuracy_over_max_d =[]
        model = rfc.RandomForestClassifier()

        for n in self.intervale_n:
            model.set_params(n_estimators = n, max_depth = self.best_param_max_d)
            model.fit(self.x_train, self.y_train)
            accuracy_over_n.append(model.score(self.x_test, self.y_test))
        for d in self.intervale_max_d:
            model.set_params(n_estimators = self.best_param_n, max_depth = d)
            model.fit(self.x_train, self.y_train)
            accuracy_over_max_d.append(model.score(self.x_test, self.y_test))

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        fig.suptitle('Accuracy over estimators')

        sns.lineplot(ax=axes[0], x=self.intervale_n,y=accuracy_over_n)
        axes[0].set_title('Accuracy over n_estimators')

        sns.lineplot(ax=axes[1],x=self.intervale_max_d, y=accuracy_over_max_d)
        axes[1].set_title('Accuracy over max_depth')

        plt.show()

    # def Visualise_n_estimators(self):
    #     CVresults = self.grid.cv_results_
    #     list_param_C=[]
        
    #     ymax = np.ones(len(CVresults["mean_test_score"]))*max(CVresults["mean_test_score"])
    #     ax1 = plt.subplot(211)
    #     ax1.plot(self.intervale_max_d, CVresults["mean_test_score"], label = 'Max Depth')
    #     ax2 = plt.subplot(221)
    #     ax2.plot(self.intervale_n, CVresults["mean_test_score"], label = 'n_estimators')
    #     plt.plot(self.intervale, ymax, label='Best value is : ' + '{:1.3f}'.format(max(CVresults["mean_test_score"])) + ' for alpha = ' + '{:1.5f}'.format(list(self.grid.best_params_.values())[0]))
    #     plt.legend()
    #     plt.xscale('log')
    #     plt.xlabel('Value of alpha')
    #     plt.ylabel('Accuracy')
    #     plt.show()