import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pandas import Series, DataFrame
from sklearn.metrics import roc_auc_score
import hvplot.pandas
import math


class ModelAnalyzer():

    def __init__(self, history: dict, resultsPath: str):
        self.history = history
        self.resultsPath = resultsPath

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
        fig.savefig(self.resultsPath + "/lstm_stacked_classification", dpi=300)

    def other(self):
        #predicted = (predicted>.2).astype(int)
        #predicted = predicted> .2
        #predictedLongs
        pass

    def roc(self, yTrue: any, yScore: any):
        return roc_auc_score(y_score=yScore, y_true=yTrue)

    def createComparisonDf(self, predictedLongsSeries: Series, predictedShortsSeries: Series,
                           actualLongs: DataFrame, actualShorts: DataFrame, longKey: str, shortKey: str) -> DataFrame:
        predictedLongs = pd.DataFrame(predictedLongsSeries, index=actualLongs.index, columns=actualLongs.columns)
        predictedShorts = pd.DataFrame(predictedShortsSeries, index=actualShorts.index, columns=actualShorts.columns)
        comparison = pd.DataFrame()
        comparison['predictedLongEntry'] = predictedLongs[longKey]
        comparison['predictedShortEntry'] = predictedShorts[shortKey]

        comparison['actualLongEntry'] = actualLongs[longKey]
        comparison['actualShortEntry'] = actualShorts[shortKey]

        comparison['predictedLongEntry'] = (comparison['predictedLongEntry'] > .80).astype(int)
        comparison['predictedShortEntry'] = (comparison['predictedShortEntry'] > .70).astype(int)

        return comparison


    def displayComparison(self, comparison: DataFrame):
        #https://stackoverflow.com/questions/43061768/plotting-multiple-scatter-plots-pandas

        #p=comparison.reset_index().plot.scatter(x='Open time', y='predictedLongEntry10', color='g')
        #comparison.reset_index().plot.scatter(x='Open time', y='actualLongEntry10', ax=p)
        plot = comparison.reset_index().hvplot(x='Open time', y=['predictedLongEntry', 'actualLongEntry'],
                                               height=600, width=1000, kind='scatter')

        hvplot.show(plot)

