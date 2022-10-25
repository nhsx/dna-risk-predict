#!/usr/bin/env python3

import os
import sys
import json
import joblib
import hashlib
import logging
import tempfile
import numpy as np
import pandas as pd
from . import utils, simulate, test, train


logger = logging.getLogger(__name__)


def train_cli(config: str):
    """ Train a model """
    config = utils.Config(config).config
    # Set global seed
    np.random.seed(config['seed'])
    # Read data and map DNA target label to 1
    data = pd.read_csv(config['input'])
    assert len(data[config['target']].unique()) == 2
    assert config['DNAclass'] in data[config['target']].unique()
    data[config['target']] = data[config['target']].apply(
        lambda x: 1 if x == config['DNAclass'] else 0)
    data = train.splitData(data, **config)
    # Train model
    models = train.trainModel(data, **config)
    # Write trained logistic and catboost models and parameter definitions
    for name, model in models.items():
        joblib.dump(model['model'], f'{config["out"]}/{name}-trained.pkl')
        with open(f'{config["out"]}/{name}-params.json', 'w') as fh:
            json.dump(model['params'], fh)
    # Write split data
    for name, df in data.items():
        df.to_pickle(f'{config["out"]}/{name}.pkl')


def test_cli(config: str):
    """ Test a pre-trained model. """
    config = utils.Config(config).config
    np.random.seed(config['seed']) # Set global seed
    data = _readData(config)
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


def retrain_cli(config: str):
    """ Re-train model on full dataset """
    config = utils.Config(config).config
    np.random.seed(config['seed']) # Set global seed
    data = _readData(config)
    modelType = config['finalModel']
    if modelType not in ['catboost', 'logistic']:
        if modelType:
            logging.error(
                f'Invalid configuration for "finalModel" - {finalModel}')
        logging.error(
            'Please set "finalModel" to either "catboost" or "logistic"')
    model = joblib.load(
        f'{config["out"]}/{modelType}-trained.pkl')
    model = train.refitData(model, data)
    joblib.dump(model, f'{config["out"]}/{modelType}-final.pkl')


def predict_cli(
        data, model, verify: bool = False, sep: str = ',', out = sys.stdout):
    data = pd.read_csv(data, sep=sep)
    model = joblib.load(model)
    data[['Attend_prob', 'DNA_prob', 'Prediction']] = (
        test.predict(model, data))
    data['Prediction'] = data['Prediction'].map({1: 'DNA', 0: 'Attend'})
    if out is not None:
        data.to_csv(out)
    if verify:
        if verifyHash(data):
            print('Success: output matches expected hash.', file=sys.stderr)
            return 0
        else:
            print('Error: output does NOT match expected hash', file=sys.stderr)
            return 1


def verifyHash(df: str, readSize: int = 4096):
    tf = tempfile.NamedTemporaryFile(delete=False)
    df.to_csv(tf.name, float_format='%.3f')
    sha256Hash = hashlib.sha256()
    with open(tf.name, 'rb') as f:
        data = f.read(readSize)
        while data:
            sha256Hash.update(data)
            data = f.read(readSize)
    hash = sha256Hash.hexdigest()
    os.remove(tf.name)
    return hash == ('801cb98641256aaf8c3a57c7af87da80'
                    'bd2f92088a8c9dcd4609c06e03395484')


def simulate_cli(
        config: str, size: int = 50_000,
        noise: float = 0.2, seed: int = 42, out=sys.stdout):
    """ Randomly generate some example data """
    assert seed > 0
    assert size > 0
    assert noise >= 0
    # Randomly generate some artificial attendance data
    df = simulate.generateData(size, seed, noise)
    df.to_csv(out, index=False)
    simulate.writeConfig(config)


def _readData(config):
    """ Read pre-split data """
    data = {}
    for name in ['X_train', 'y_train', 'X_test', 'y_test', 'X_val', 'y_val']:
        data[name] = pd.read_pickle(f'{config["out"]}/{name}.pkl')
    return data
