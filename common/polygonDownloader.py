from typing import List

from vectorbtpro.data.base import DataT
import vectorbtpro as vbt
import datetime
import dateutil.relativedelta

from configs.configSecrets import ConfigSecrets


def downloadData(symbols: List[str], startDate: str, endDate: str, timeFrame: str ) -> DataT:
    vbt.PolygonData.set_custom_settings(
        client_config=dict(
            api_key=ConfigSecrets.POLYGON_API_KEY
        )
    )
    '''
    allDataStart: str = '2017-01-01 UTC'
    allDataEnd: str = '2023-01-01 UTC'
    allDataTimeFrame: str = '1h'
    '''
    data = vbt.PolygonData.fetch(
        symbols,
        start=startDate,
        end=endDate,
        timeframe=timeFrame
    )
    return data

def downloadMonthHourly(symbols):
    now = datetime.datetime.now()
    lastMonth = now + dateutil.relativedelta.relativedelta(months=-1)
    tomorrow = now + dateutil.relativedelta.relativedelta(days=+2)
    tmStr = tomorrow.strftime("%Y-%m-%d") + " UTC"
    lstStr = lastMonth.strftime("%Y-%m-%d") + " UTC"
    return downloadData(symbols, lstStr, tmStr, '1h')
