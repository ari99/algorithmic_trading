from pandas import DataFrame
from vectorbtpro import Portfolio, Ranges
from vectorbtpro._typing import SeriesFrame, MaybeTuple

from common.labelMaker import LabelMaker


class CleanEntriesExits:

    def __init__(self, allData: DataFrame, period: int):

        shortExitsLabel = LabelMaker.makeLongEntryLabel(period) #'longEntry'+str(period)
        shortEntriesLabel = LabelMaker.makeShortEntryLabel(period)
        # the short entries are the long exits
        longExitsLabel = LabelMaker.makeShortEntryLabel(period)
        longEntriesLabel = LabelMaker.makeLongEntryLabel(period)

        longEntries = allData[longEntriesLabel].copy()
        longExits = allData[longExitsLabel].copy()
        shortEntries = allData[shortEntriesLabel].copy()
        shortExits = allData[shortExitsLabel].copy()

        longs: MaybeTuple[SeriesFrame] = longEntries.vbt.signals.clean(longExits)
        shorts: MaybeTuple[SeriesFrame] = shortEntries.vbt.signals.clean(shortExits)
        self.cleanLongEntries: SeriesFrame = longs[0]
        self.cleanLongExits: SeriesFrame = longs[1]
        self.cleanShortEntries: SeriesFrame = shorts[0]
        self.cleanShortExits: SeriesFrame = shorts[1]

    def getLongRanges(self) -> Ranges:
        ranges: Ranges = self.cleanLongEntries.vbt.signals.between_ranges(
            other= self.cleanLongExits,
            from_other=True
        )
        # ranges.avg_duration
        return ranges

    def getShortRanges(self) -> Ranges:
        ranges: Ranges = self.cleanShortEntries.vbt.signals.between_ranges(
            other= self.cleanShortExits,
            from_other=True
        )
        # ranges.avg_duration
        return ranges

    def getLongStats(self):
        return self.cleanLongEntries.vbt.signals.stats(settings=dict(other=self.cleanLongExits))

    def getShortStats(self):
        return self.cleanShortEntries.vbt.signals.stats(settings=dict(other=self.cleanShortExits))


