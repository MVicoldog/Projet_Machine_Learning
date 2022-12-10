
import matplotlib.pyplot as plt
import numpy as np


class gaussianNB_visualizer:

    def __init__(self, grid, intervale):
        self.grid = grid
        self.intervale = intervale

    def Visualise_tuning(self):
        CVresults = self.grid.cv_results_

        ymax = np.ones(len(CVresults["mean_test_score"])) * \
            max(CVresults["mean_test_score"])
        plt.figure(figsize=(10, 10))
        plt.plot(self.intervale, ymax, label='Best value is : ' + '{:1.3f}'.format(max(
            CVresults["mean_test_score"])) + ' for var = ' + '{:1.5f}'.format(list(self.grid.best_params_.values())[0]))
        plt.plot(self.intervale, CVresults["mean_test_score"])
        plt.legend()
        plt.title(
            'Variation du score en fonction de l"hyper-paramètre var_smoothing')
        plt.xscale('log')
        plt.xlabel('var_smoothing')
        plt.ylabel('Accuracy')
        plt.show()
