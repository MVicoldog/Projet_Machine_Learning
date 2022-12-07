
import matplotlib.pyplot as plt
import numpy as np
from visualizers import learning_curve as lc

class ridge_visualizer:

    def __init__(self, grid, intervale):
        self.grid = grid
        self.intervale = intervale
    
    def Visualise_tuning(self):
        CVresults = self.grid.cv_results_
       
        ymax = np.ones(len(CVresults["mean_test_score"]))*max(CVresults["mean_test_score"])
        plt.figure(figsize=(10,10))
        plt.plot(self.intervale, ymax, label='Best value is : ' + '{:1.3f}'.format(max(CVresults["mean_test_score"])) + ' for alpha = ' + '{:1.5f}'.format(list(self.grid.best_params_.values())[0]))
        
        
        plt.plot(self.intervale, CVresults["mean_test_score"])
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Value of alpha')
        plt.ylabel('Accuracy')
        plt.show()