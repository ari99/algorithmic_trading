from pandas import Series, DataFrame


class FlipBarData:
    def flipBarData(self, df: DataFrame) -> DataFrame:
        newBarData: DataFrame = df.copy()
        adjust: int = df.High.max() + 100
        newClose: Series = self.flipSeries(df.Close, adjust)
        newOpen: Series = self.flipSeries(df.Open, adjust)
        newHigh: Series = self.flipSeries(df.High, adjust)
        newLow: Series = self.flipSeries(df.Low, adjust)
        newBarData['Close'] = newClose
        newBarData['Open'] = newOpen
        newBarData['High'] = newHigh
        newBarData['Low'] = newLow

        #self.barData.data[symbol]['Close'].plot()
        #self.barData.data[symbol]['Open'].plot()
        return newBarData

    # the adjust params separates the two series so the flipped series doesnt go negative
    def flipSeries(self, originalSeries: Series, adjust: int) -> Series:
        tempDf: DataFrame = DataFrame()
        tempDf['originalSeries'] = originalSeries
        tempDf['diff'] = originalSeries.diff()
        tempDf['negdiff'] = tempDf['diff'] * -1
        start = tempDf['originalSeries'].iloc[0]
        tempDf['newSeries'] = 0
        tempDf.loc[tempDf.index[0], 'newSeries'] = start
        for i in range(1, len(tempDf)):
            tempDf.loc[tempDf.index[i], 'newSeries'] = \
                tempDf.loc[tempDf.index[i-1], 'newSeries'] + tempDf.loc[tempDf.index[i], 'negdiff']
        tempDf['newSeries'] = tempDf['newSeries']+adjust
        return tempDf['newSeries']

