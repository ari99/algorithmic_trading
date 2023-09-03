from typing import List
from pandas import DataFrame


class CorrelationMaker:

    def prepareCorrelationInputData(self, allData: DataFrame, dontDrop: str) -> DataFrame:
        data: DataFrame = allData.loc[:, :]
        data = data.reset_index()
        cols: List[str] = [
                'Open time',
                'longEntry10', 'shortEntry10',
                'longEntry20', 'shortEntry20',
                'longEntry50',
                'shortEntry50',  'longEntry100',
                'shortEntry100'
                ]
        cols = list(filter(lambda c: c != dontDrop, cols))

        data = data.drop(columns=cols)
        data[dontDrop] = data[dontDrop].astype(int)
        data['outWQA95'] = data['outWQA95'].astype(int)
        data['outWQA61'] = data['outWQA61'].astype(int)
        data['outWQA75'] = data['outWQA74'].astype(int)
        # https://stackoverflow.com/questions/63550024/seaborn-clustermap-floatingpointerror-nan-dissimilarity-value
        fillMean = lambda col: col.fillna(col.mean())
        data = data.apply(fillMean, axis=0)

        return data

    def makeCorrelationData(self, preparedData: DataFrame, target: str, minCor: float):

        preparedData = preparedData.loc[:, (preparedData != 0).any(axis=0)]
        cor = preparedData.corr('spearman')
        # cor = cor.loc[:, (cor != 0).any(axis=0)]
        cor = cor.dropna(axis='index', thresh=2)
        cor = cor.dropna(axis='columns', thresh=2)
        cor = cor.dropna(axis='columns', thresh=1)
        cor = cor.dropna(axis='index', thresh=1)

        cor = cor.loc[:, (cor != 0).any(axis=0)]

        cor = cor[(cor[target] > minCor) | (cor[target] < -1*minCor)]

        # https://stackoverflow.com/questions/57464432/drop-columns-in-pandas-dataframe-based-on-conditions
        cor = (cor.T
               .loc[lambda x: ((x[target] > minCor) | (x[target] < -1*minCor))]
               .T.reset_index().set_index('index'))

        return cor

