from numpy import ndarray
from pandas import DataFrame
from vectorbtpro import Portfolio
import numpy as np
from pandas import Series
from numpy.random import default_rng


class WhitesCheck:

    def run(self, pf: Portfolio, close: Series) -> None:
        alPctCloseDetr: Series = self.createAlPctCloseDetr(close, pf.allocations)
        originalMean = alPctCloseDetr.mean()
        zeroed = alPctCloseDetr - alPctCloseDetr.mean()
        samplingInds: ndarray[int] = self.createRandomIndexes(zeroed, 5000)
        samplesMeans = self.createSampleMeans(samplingInds, zeroed)
        print("originalMean is " + str(originalMean) + " count random greater " +
              str(samplesMeans[samplesMeans > originalMean].count() / 5000)
              + " random max: " + str(samplesMeans.max()))
        self.printHistogram(samplesMeans, originalMean)

    def createSampleMeans(self, samplingInds: ndarray[int], zeroed: Series):
        def col_p_val(indexes_col):
            # print((zeroed.values[indexes_col]).shape)
            samples = zeroed.values[indexes_col].mean()
            return samples
        indexes = DataFrame(samplingInds)
        means = indexes.apply(col_p_val, axis=0)
        return means

    def createRandomIndexes(self, zeroed: Series, num) -> ndarray[int]:
        # n = 5000
        samplingInds: ndarray[int] = np.random.randint(0, len(zeroed), size=(len(zeroed), num))
        return samplingInds

    def createAlPctCloseDetr(self, close: Series, allocations: Series) -> Series:
        allocations = np.sign(allocations)
        pctCloseDetrended = close.pct_change() - close.pct_change().mean()
        alPctCloseDetr: Series = allocations * pctCloseDetrended
        alPctCloseDetr = alPctCloseDetr.iloc[1:]
        return alPctCloseDetr

    def printHistogram(self, randomReturns, meanValue) -> None:
        bins = [x for x in np.arange(-.0002, .00025, .00001)]
        #a.plot.hist(bins=bins, column=["TotalReturn"])
        patch_index = np.digitize([meanValue], bins)[0]

        p = randomReturns.plot.hist(bins=bins)
        p.patches[patch_index].set_color('orange')