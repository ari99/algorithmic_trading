import pandas as pd
from pandas import Series, DataFrame

from common.checkpointHandler import CheckpointHandler
from configs.config import Config


class DailyReturns(CheckpointHandler):
    def __init__(self, closeSeries: Series):
       self.closeSeries = closeSeries

    def createData(self) -> DataFrame:
        data: DataFrame = self.createDailyReturns()
        return data

    def createDailyReturns(self) -> DataFrame:
        dailyReturnsPct = (self.closeSeries.to_frame()
                           .resample('D').last().pct_change().dropna(axis=0)
                           .sort_index(ascending=False)
                           )
        n = len(dailyReturnsPct)
        daily_window_size = Config.dailyReturnsWindowSize  # period
        tcols = list(range(daily_window_size))
        tickers = dailyReturnsPct.columns
        dailyData: DataFrame = pd.DataFrame()

        for i in range(n-daily_window_size-1):
            # this removes the first row from the df
            df = dailyReturnsPct.iloc[i:i+daily_window_size+1]
            date = df.index.max()
            dailyData = pd.concat([dailyData, df.reset_index(drop=True).T.assign(date=date).set_index('date')])

        dailyData[tcols] = (dailyData[tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                                    upper=x.quantile(.99))))
        dailyData = dailyData.rename(columns={0: 'fwd_returns'})
        dailyData['label'] = (dailyData['fwd_returns'] > 0).astype(int)
        dailyData = dailyData.sort_index()
        return dailyData





