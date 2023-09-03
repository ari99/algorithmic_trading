import pandas as pd
from pandas import Series, DataFrame

from common.checkpointHandler import CheckpointHandler
from configs.config import Config

''' 
TODO: add this:
sequence = list(range(1, Config.hourlyReturnsWindowSize+1))
trainLaggedReturns = returns.loc[:, sequence].values.reshape(-1, Config.hourlyReturnsWindowSize , 1)
'''


class HourlyReturns(CheckpointHandler):

    def __init__(self, closeSeries: Series):
        self.closeSeries = closeSeries

    def createData(self) -> DataFrame:
        data = self.createHourlyReturns()
        return data

    def createHourlyReturns(self) -> DataFrame:
        returnsPct = (self.closeSeries
                      .pct_change()
                      .sort_index(ascending=False))
        n = len(returnsPct)
        windowSize = Config.hourlyReturnsWindowSize  # period
        hourlyData: DataFrame = pd.DataFrame()
        for i in range(n-windowSize-1):
            # this removes the first row from the df
            df = returnsPct.iloc[i:i+windowSize+1].to_frame()
            date = df.index.max()
            hourlyData = pd.concat([hourlyData, df.reset_index(drop=True).T.assign(date=date).set_index('date')])

        hourlyData = hourlyData.rename(columns={0: 'fwd_returns'}).sort_index().dropna()
        hourlyData['fwd_returns_label'] = (hourlyData['fwd_returns'] > 0).astype(int)

        return hourlyData


#   def createDailyReturns(self):




