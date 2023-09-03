import pandas as pd
from pandas import Series, DataFrame
import hvplot.pandas


class PredictComparer:

    def createComparisonDf(self, predictedLongsSeries: Series, predictedShortsSeries: Series,
                           actualLongs: DataFrame, actualShorts: DataFrame, longKey: str, shortKey: str) -> DataFrame:

        predictedLongs = pd.DataFrame(predictedLongsSeries, index=actualLongs.index, columns=actualLongs.columns)
        predictedShorts = pd.DataFrame(predictedShortsSeries, index=actualShorts.index, columns=actualShorts.columns)
        comparison: DataFrame = pd.DataFrame()
        comparison['predictedLongEntry'] = predictedLongs[longKey]
        comparison['predictedShortEntry'] = predictedShorts[shortKey]
        comparison['actualLongEntry'] = actualLongs[longKey]
        comparison['actualShortEntry'] = actualShorts[shortKey]
        comparison['predictedLongEntry'] = (comparison['predictedLongEntry'] > .80).astype(int)
        comparison['predictedShortEntry'] = (comparison['predictedShortEntry'] > .70).astype(int)

        return comparison

    def createAllComparisonDf(self, predictedLongsSeries: Series, predictedShortsSeries: Series,
                              actualLongs: DataFrame, actualShorts: DataFrame) -> DataFrame:
        comparison: DataFrame = pd.DataFrame()
        comparison.index = actualLongs.index
        comparison['predictedLongEntry'] = predictedLongsSeries
        comparison['predictedShortEntry'] = predictedShortsSeries

        comparison['actualLongEntry'] = actualLongs.values
        comparison['actualShortEntry'] = actualShorts.values
        for i in [x / 100.0 for x in range(0, 100, 5)]:
            comparison['predictedLongEntry_'+str(i)] = (comparison['predictedLongEntry'] > i).astype(int)
            comparison['predictedShortEntry_'+str(i)] = (comparison['predictedShortEntry'] > i).astype(int)

        return comparison

    def displayComparison(self, comparison: DataFrame) -> None:
        #https://stackoverflow.com/questions/43061768/plotting-multiple-scatter-plots-pandas

        #p=comparison.reset_index().plot.scatter(x='Open time', y='predictedLongEntry10', color='g')
        #comparison.reset_index().plot.scatter(x='Open time', y='actualLongEntry10', ax=p)
        plot = comparison.reset_index().hvplot(x='Open time', y=['predictedLongEntry', 'actualLongEntry'],
                                               height=600, width=1000, kind='scatter')

        hvplot.show(plot)

