
import controllers.gaussianNB_classifier_controller as gNBcc
import matplotlib.pyplot as plt
import numpy as np

class gaussianNB_visualizer:

    def __init__(self, grid, intervale):
        self.grid = grid
        self.intervale = intervale
    
    def Visualise(self):
        CVresults = self.grid.cv_results_
        list_param_C=[]
        for i in range(0,len(CVresults["params"])):
            list_param_C.append(list(CVresults["params"][i].values())[0])
        ymax = np.ones(len(CVresults["mean_test_score"]))*max(CVresults["mean_test_score"])
        plt.figure(figsize=(10,10))
        plt.plot(self.intervale, ymax, label='Best value is : ' + '{:1.3f}'.format(max(CVresults["mean_test_score"])) + ' for var = ' + '{:1.5f}'.format(list(self.grid.best_params_.values())[0]))
        
        
        plt.plot(self.intervale, CVresults["mean_test_score"])
        plt.plot(self.intervale, ymax, label='Best value is : ' + '{:1.3f}'.format(max(CVresults["mean_test_score"])) + ' for var = ' + '{:1.5f}'.format(list(self.grid.best_params_.values())[0]))
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Value of variance')
        plt.ylabel('Accuracy')
        plt.show()