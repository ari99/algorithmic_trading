from typing import List
import vectorbtpro as vbt
from configs.config import Config

from pandas import DataFrame
from common.featuresTargetsUnshiftedMaker import FeaturesTargetsUnshiftedMaker


class FeaturePreparer():

    def makeFeatureDf(self, allData: DataFrame) -> DataFrame:
        modelFeatures: DataFrame = vbt.HDFData.fetch(Config.relativeTestFeaturesPath).data['test_features']
        modelFeaturesColumns: List[str] = modelFeatures.columns.tolist()

        targetColumns: List[str] = FeaturesTargetsUnshiftedMaker.targetColumnNames
        featureDataShifted: DataFrame = allData.copy()
        featureDataShifted.insert(0, "CurrentOpen",  allData['Open'])
        featureDataShifted = self.shiftFeatures(targetColumns, featureDataShifted)
        dontRemove = ["CurrentOpen"]
        dontRemove.extend(targetColumns)
        dontRemove.extend(modelFeaturesColumns)
        # remove all nans columns
        featureDataShifted = featureDataShifted.dropna(axis=1, how='all')
        # remove all 0's columns
        zerocolumns = (featureDataShifted.loc[:, (featureDataShifted == 0).all(axis=0)]).columns.tolist()
        print("Zeros")
        print(zerocolumns)
        print("Zeros length " + str(len(zerocolumns)))
        toRemove = []
        for col in zerocolumns:
            if col not in dontRemove:
                toRemove.append(col)
        print("To remove")
        print(toRemove)
        print("toRemove length " + str(len(toRemove)))

        featureDataShifted = featureDataShifted.drop(
            columns=[col for col in featureDataShifted if col in toRemove])
        # remove columns that werent in the columns the model was trained on
        featureDataShifted = featureDataShifted.drop(
            columns=[col for col in featureDataShifted if col not in dontRemove])

        #featureDataShifted = featureDataShifted.loc[:, (featureDataShifted != 0).any(axis=0)]

        featureDataShifted = self.convertBooleans(featureDataShifted, targetColumns)
        fillMean = lambda col: col.fillna(col.mean())
        featureDataShifted = featureDataShifted.apply(fillMean, axis=0)

        return featureDataShifted


    def convertBooleans(self, featureDataShifted: DataFrame, targetColumns: List[str]) -> DataFrame:
        convert = ["outWQA95", "outWQA61"]
        convert.extend(targetColumns)
        currentColumns = featureDataShifted.columns.tolist()
        for column in convert:
            if column in currentColumns:
                featureDataShifted[column] = featureDataShifted[column].astype(int)

        return featureDataShifted

    def shiftFeatures(self, targetColumns: List[str], featureDataShifted: DataFrame):
        dontShift = ["CurrentOpen"]
        dontShift.extend(targetColumns)
        mask = ~(featureDataShifted.columns.isin(dontShift))

        cols_to_shift = featureDataShifted.columns[mask]
        shiftReturned = featureDataShifted.loc[:, mask].shift(1)
        print("cols to shift " + str(len(cols_to_shift))
              + " shiftedData " + str(len(shiftReturned.columns))
              + " shape " + str(featureDataShifted.shape)
              )
        print(featureDataShifted.columns[featureDataShifted.columns.duplicated(keep=False)])
        # to test whether its masking the correct thing you can set =0
        featureDataShifted[cols_to_shift] = featureDataShifted.loc[:, mask].shift(1)

        # remove the first row because it will be all NaNs after shifting
        featureDataShifted = featureDataShifted.iloc[1:, :]
        return featureDataShifted

