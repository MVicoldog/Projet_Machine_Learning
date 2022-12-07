
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

class random_forest_visualizer:

    def __init__(self, grid, intervale_n, intervale_max_d):
        self.grid = grid
        self.intervale_n = intervale_n
        self.intervale_max_d = intervale_max_d

    # def Visualise(self):
    #     CVresults = self.grid.cv_results_
    #     fig = px.scatter_3d( CVresults["mean_test_score"] ,x= np.arange(0, len(CVresults["params"])),
    #                          y=self.intervale_n, z=self.intervale_max_d)
    #     fig.show()
    
    # def Visualise(self):
    #     CVresults = self.grid.cv_results_
    #     list_param_C=[]
    #     for i in range(0,len(CVresults["params"])):
    #         list_param_C.append(list(CVresults["params"][i].values())[0])
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