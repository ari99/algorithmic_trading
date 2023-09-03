
from common.polygonDownloader import downloadMonthHourly
import logging
log = logging.getLogger(__name__)
#https://pypi.org/project/schedule/

import schedule
import time
from .predictor import Predictor, Prediction
from .provider import Provider
from .alpacaProvider import AlpacaProvider
from .trader import Trader


def job():
    print("Starting scheduled job")
    predictor = Predictor()
    latest= downloadMonthHourly(['X:BTCUSD'])
    symbol= 'X:BTCUSD'
    longTarget = 'longEntry50'
    shortTarget = 'shortEntry50'
    targetPeriod = 50
    prediction: Prediction = predictor.makePrediction(latest, symbol, longTarget, shortTarget, targetPeriod)
    provider: Provider = AlpacaProvider()
    trader: Trader = Trader(provider)
    trader.doTrade(prediction)

def runSchedule():
    print("-----starting schedule")
    # run every hour at the 15th minute
    schedule.every().hour.at(":15").do(job)

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    job()