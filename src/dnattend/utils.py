#!/usr/bin/env python3

import yaml
import pprint
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def setVerbosity(
        level=logging.INFO, handler=logging.StreamHandler(),
        format='%(name)s - %(levelname)s - %(message)s'):
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    pkg_logger = logging.getLogger('dnattend')
    pkg_logger.setLevel(level)
    pkg_logger.addHandler(handler)


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


class Config():
    """ Custom class to read, validate and
        set defaults of YAML configuration file
        """

    def __init__(self, pathToYaml):
        self.error = False
        # Reserved string for mandatory arguments
        self.mandatory = 'mandatory'
        self.pathToYaml = pathToYaml
        self.config = self._readYAML()
        self._setDefault(self.config, self.default)
        self._postProcessConfig()
        if self.error:
            logging.error(
                f'Expected format:\n\n{pprint.pformat(self.default)}')
            raise ValueError('Invalid configuration.')

    def _setDefault(self, config, default, path=''):
        """ Recursively set default values. """
        for k in default:
            if isinstance(default[k], dict):
                if not isinstance(config[k], dict):
                    logging.error(
                        f'"{config[k]}" should be a dictionary.')
                    self.error = True
                else:
                    self._setDefault(
                        config.setdefault(k, {}), default[k], path=path+k)
            else:
                if (((k not in config) or (config[k] is None))
                        and (default[k] == self.mandatory)):
                    msg = f'{path}: {k}' if path else k
                    logging.error(
                        f'Missing mandatory config "{msg}".')
                    self.error = True
                config.setdefault(k, default[k])

    def _readYAML(self):
        """ Custom validation """
        with open(self.pathToYaml, 'r') as stream:
            return yaml.safe_load(stream)

    @property
    def default(self):
        """ Default values of configuration file. """
        return ({
            'input': self.mandatory,
            'target': self.mandatory,
            'out': '.',
            'hyperParams': None,
            'catCols': [],
            'boolCols': [],
            'numericCols': [],
            'train_size': 0.7,
            'test_size': 0.15,
            'val_size': 0.15,
            'tuneThresholdBy':     'f1',
            'cvFolds':             5,
            'catboostIterations':  100,
            'hypertuneIterations': 5,
            'evalIterations':      10_000,
            'earlyStoppingRounds': 10,
            'seed':                42
        })


    def _postProcessConfig(self):
        """ Additional config modifications """
        pass
