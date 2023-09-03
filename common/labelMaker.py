from typing import List


class LabelMaker():

    @classmethod
    def makeLongEntryLabel(cls, period: int) -> str:
        return "longEntry"+str(period)

    @classmethod
    def makeLongEntryLabelsList(cls, periods: List[int]) -> List[str]:
        results = []
        for period in periods:
            results.append("longEntry"+str(period))
        return results

    @classmethod
    def makeShortEntryLabel(cls, period: int) -> str:
        return "shortEntry"+str(period)

    @classmethod
    def makeShortEntryLabelsList(cls, periods: List[int]) -> List[str]:
        results = []
        for period in periods:
            results.append("shortEntry"+str(period))
        return results

    @classmethod
    def makeAllEntryLabelsList(cls, periods: List[int]) -> List[str]:
        results = []
        for period in periods:
            results.append("longEntry"+str(period))
            results.append("shortEntry"+str(period))

        return results

