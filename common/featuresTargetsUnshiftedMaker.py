import vectorbtpro as vbt
import pandas as pd
from pandas import DataFrame
import numpy as np
from vectorbtpro.data.custom import HDFData
from common.labelMaker import LabelMaker
idx = pd.IndexSlice


class FeaturesTargetsUnshiftedMaker:
    periods = [10, 20, 50, 100]
    targetColumnNames = LabelMaker.makeAllEntryLabelsList(periods)

    def loadData(self, barData: HDFData, symbol: str):
        taFeatures: DataFrame = self.createTAFeatures(barData.select(symbol))
        taFeatures = taFeatures.dropna(axis=1, how='all')
        symbolData: DataFrame = self.createSymbolDF(taFeatures, barData.data[symbol])
        symbolData = self.addAllLongShorts(symbolData)

        return symbolData

    def addAllLongShorts(self, allData: DataFrame) -> DataFrame:
        for period in self.periods:
            allData = self.addLongShortsTargets(allData, period)

        return allData

    def createSymbolDF(self, taFeatures: DataFrame, barData: DataFrame) -> DataFrame:
        result = barData.join(taFeatures, how="inner")
        return result

    def createTAFeatures(self, data: HDFData) -> DataFrame:
        print("Data shape " + str(data.shape))
        print("Data type " + str(type(data.shape)))

        taData: DataFrame = data.run("talib", periods=vbt.run_func_dict(mavp=14))
        print("taData shape " + str(taData.shape))

        taData = taData.sort_index(axis=1)

        # these have "inf" values after going through featureMaker
        taData = taData.drop(('exp', 'real'), axis=1)
        taData = taData.drop(('cosh', 'real'), axis=1)
        taData = taData.drop(('sinh', 'real'), axis=1)

        tuples = []
        for a in taData.columns.to_flat_index():
            if len(a) == 3:
                tuples.append((a[0]+"_"+a[1], a[2])) #when the hdf contains multiple symbols its 3
            elif len(a) == 2:
                tuples.append((a[0]+"_"+a[1]))
            else:
                print("unknown length for ta columns- check this")

        taData.columns = tuples

        noWQA = [48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 84, 87, 89, 90, 91, 93, 97, 100]
        for i in range(1, 102):
            if i not in noWQA:
                outWQA = vbt.wqa101(i).run(high=data.high, low=data.low, open=data.open,
                                           close=data.close, volume=data.volume).out
                columnName = "outWQA" + str(i)
                df = outWQA.to_frame()
                df.columns = [columnName]
                print("taData shape " + str(taData.shape) + " outWQA " + str(outWQA.shape))
                taData = taData.join(df, how="inner")

        return taData

    def addLongShortsTargets(self, df: DataFrame, period: int) -> DataFrame:
        longLabel: str = LabelMaker.makeLongEntryLabel(period)
        shortLabel: str = LabelMaker.makeShortEntryLabel(period)

        minmax = df.groupby(np.arange(len(df.index)) // period)['Close'].agg(['idxmin', 'idxmax'])
        minmax[longLabel] = np.where(minmax['idxmax'] > minmax['idxmin'], True, False)
        minmax[shortLabel] = np.where(minmax['idxmin'] > minmax['idxmax'], True, False)

        df: DataFrame = df.merge(minmax.set_index('idxmin').drop(columns=['idxmax']), left_index=True, right_index=True,
                      how='left', suffixes=('_min', '_min'))
        df = df.merge(minmax.set_index('idxmax').drop(columns=['idxmin']), left_index=True, right_index=True,
                      how='left', suffixes=('_min', '_max'))
        minLongLabel = longLabel+"_min"
        minShortLabel = shortLabel+"_min"
        maxLongLabel = longLabel+"_max"
        maxShortLabel = shortLabel+"_max"

        df[longLabel] = df[minLongLabel].fillna(df[maxLongLabel])
        df[shortLabel] = df[minShortLabel].fillna(df[maxShortLabel])
        df[longLabel] = df[longLabel].fillna(False)
        df[shortLabel] = df[shortLabel].fillna(False)

        df = df.drop(columns=[minLongLabel, minShortLabel, maxLongLabel, maxShortLabel])
        return df








