#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, TransformerMixin

class _prepareData(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            catCols: list = None,
            numericCols: list = None,
            boolCols: list = None,
            decisionThreshold: float = 0.5):
        self.catCols = [] if catCols is None else catCols
        self.numericCols = [] if numericCols is None else numericCols
        self.boolCols = [] if boolCols is None else boolCols
        # Store parameter for decision threshold within model object
        self.decisionThreshold = decisionThreshold
        self._setCatColIdx()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in self.boolCols:
            X[col] = self._mapBoolCol(X[col])
        X[self.catCols] = X[self.catCols].astype(str)
        return X.loc[:, self.validCols]

    def _mapBoolCol(self, col):
        col = col.apply(lambda x: str(x).lower().strip()[0])
        names = ({
            '1': 1, 'y': 1, 't': 1,
            '0': 0, 'n': 0, 'f': 0
        })
        col = col.map(names)
        col.loc[~col.isin([0,1])] = np.nan
        return col

    @property
    def validCols(self):
        return self.catCols + self.numericCols + self.boolCols

    def _setCatColIdx(self):
        """ Get indices of categoric cols """
        self.catColIdx = []
        for col in self.catCols:
            if col in self.validCols:
                self.catColIdx.append(
                    self.validCols.index(col))


def _tuneThreshold(model, X_train, y_train):
    y_trainInt = y_train.apply(lambda x: 1 if x == model.classes_[1] else 0)
    trainPredictProb = model.predict_proba(X_train)[:,1]

    fpr, tpr, thresholds = roc_curve(
        y_trainInt, trainPredictProb, drop_intermediate=False)

    idx = np.argmin(np.abs(fpr + tpr - 1))
    optimalThreshold = thresholds[idx]
    return optimalThreshold
