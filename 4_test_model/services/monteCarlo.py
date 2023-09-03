from vectorbtpro import Portfolio
import numpy as np
from pandas import Series
from numpy.random import default_rng


class MonteCarloCheck:

    def run(self, pf: Portfolio, close: Series) -> None:
        allocations: Series = self.createAllocation(pf)
        pctCloseDetrended: Series = self.createDetrendedPctClose(close)
        randomReturns = self.createRandomReturns(pctCloseDetrended, allocations)
        meanReturn = (pctCloseDetrended * allocations).mean()
        print("meanReaturn is " + str(meanReturn) + " count random greater " +
              str(randomReturns[randomReturns > meanReturn].count()) + " random max: " + str(randomReturns.max()))
        pValue = randomReturns[randomReturns > meanReturn].count() / 10000
        print("pValue is " + str(pValue))
        self.printHistogram(randomReturns, meanReturn)

    def createAllocation(self, pf: Portfolio) -> Series:
        allocations = np.sign(pf.allocations)
        allocations = allocations.iloc[1:]
        return allocations

    def createDetrendedPctClose(self, close: Series) -> Series:
        pctCloseDetrended: Series = close.pct_change() - close.pct_change().mean()
        pctCloseDetrended = pctCloseDetrended.iloc[1:]
        return pctCloseDetrended

    def createRandomReturns(self, pctCloseDetrended: Series, allocations: Series) -> Series:
        print("shape of detrended close to create random " + str(pctCloseDetrended.shape))
        rng = default_rng()
        indexes = rng.choice(len(pctCloseDetrended), size=len(pctCloseDetrended), replace=False)
        randomReturns = Series()
        for i in range(0,20000):
            mcrIndexes = rng.choice(len(pctCloseDetrended), size=len(pctCloseDetrended), replace=False)
            randomCloses = pctCloseDetrended.values[mcrIndexes]
            alSamples = allocations * randomCloses
            randomReturns.at[i]= alSamples.mean()
        return randomReturns

    def printHistogram(self, randomReturns, meanValue) -> None:
        bins = [x for x in np.arange(-.0002, .00025, .00001)]
        #a.plot.hist(bins=bins, column=["TotalReturn"])
        patch_index = np.digitize([meanValue], bins)[0]

        p = randomReturns.plot.hist(bins=bins)
        p.patches[patch_index].set_color('orange')