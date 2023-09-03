import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import math

class ModelAnalyzer():

    def __init__(self, history: dict):
        self.history = history

    def makeFigs(self):
        def which_metric(m):
            return m.split('_')[-1]

        loss_history = pd.DataFrame(self.history)

        fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(18,40))
        for i, (metric, hist) in enumerate(loss_history.groupby(which_metric, axis=1)):
            row = math.floor(i / 2)
            col = math.floor(i % 2)
            hist.plot(ax=axes[row][col], title=metric)
            axes[row][col].legend(['Training', 'Validation'])

        sns.despine()
        fig.tight_layout()
        #fig.savefig(self.resultsPath + "/lstm_stacked_classification", dpi=300)

    def roc(self, yTrue: any, yScore: any):
        return roc_auc_score(y_score=yScore, y_true=yTrue)
