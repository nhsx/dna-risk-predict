#!/usr/bin/env python3

import sys
import yaml
import json
import joblib
import pandas as pd
from . import utils, simulate, test, train


def train_cli(config: str):
    """ Run training component """
    config = utils.Config(config).config
    data = pd.read_csv(config['input'])
    data = train.splitData(data, **config)
    models = train.trainModel(data, **config)
    for name, model in models.items():
        joblib.dump(model['model'], f'{config["out"]}/{name}-trained.pkl')
        with open(f'{config["out"]}/{name}-params.json', 'w') as fh:
            json.dump(model['params'], fh)


def test_cli(config: str):
    """ Run training component """
    config = utils.Config(config).config
    data = pd.read_csv(config['input'])
    data = train.splitData(data, **config)
    models = {'catboost': {}, 'logistic': {}}
    for name in models:
        models[name]['model'] = joblib.load(
            f'{config["out"]}/{name}-trained.pkl')
        with open(f'{config["out"]}/{name}-params.json') as fh:
            models[name]['params'] = json.load(fh)

    featureImportances = test.getFeatureImportance(models['catboost']['model'])
    fig = featureImportances.plot.barh()
    fig.figure.savefig(f'{config["out"]}/featureImportances.png', dpi=300)

    fig, ax = test.plotROC(models, data)
    fig.figure.savefig(f'{config["out"]}/ROCcurve.png', dpi=300)

    fig, ax = test.plotPrecisionRecall(models, data)
    fig.figure.savefig('PRcurve.png', dpi=300)

    fig, ax = test.plotCalibrationCurve(models, data, strategy='quantile')
    fig.figure.savefig('CalibrationCurve.png', dpi=300)

    for name in models:
        report = test.evaluate(models[name]['model'], data)
        with open(f'{config["out"]}/{name}-report.json', 'w') as fh:
            json.dump(report, fh)


def simulate_cli(config: str, size: int, noise: float, seed: int):
    """ Randomly generate some example data """
    assert seed > 0
    assert size > 0
    assert noise >= 0
    # Randomly generate some artificial attendance data
    df = simulate.generateData(size, seed, noise)
    df.to_csv(sys.stdout, index=False)
    _writeConfig(config)


def _writeConfig(config: str = None):
    catCols = ['day', 'priority', 'speciality', 'consultationMedia', 'site']
    boolCols = ['firstAppointment']
    numericCols = ['age']
    config_settings = ({
        'input': 'DNAttend-example.csv',
        'target': 'status',
        'catCols': catCols,
        'boolCols': boolCols,
        'numericCols': numericCols,
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
    if config is None:
        yaml.dump(config_settings, sys.stderr)
    else:
        with open(config, 'w') as fh:
            yaml.dump(config_settings, fh)
