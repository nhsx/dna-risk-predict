#!/usr/bin/env python3

""" Simulate some dummy data for testing """

import yaml
import random
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _setProb(x: pd.Series, noise: float = 0.2) -> pd.Series:
    """ Simulate DNA probability with artificial values """
    modifiers = ({
        'day': 0.05,
        'priority': 0.1,
        'firstAppointment': 0.2,
        'consultationMedia': 0.3,
        'speciality': 0.35,
        'site': 0.4,
        'noise': noise
    })
    maxModifier = np.array(list(modifiers.values())).sum() + 0.5
    minModifier = -maxModifier
    if x['day'] in ['Saturday', 'Sunday']:
        modifiers['day'] *= -1
    if x['priority'] == 'Two Week Wait':
        modifiers['priority'] *= -1
    if not x['firstAppointment']:
        modifiers['firstAppointment'] *= -1
    if x['consultationMedia'] == 'In-Person':
        modifiers['consultationMedia'] *= -1
    if x['speciality'] == 'Audiology':
        modifiers['speciality'] *= -1
    if x['site'] == 'Lakeside':
        modifiers['site'] *= -1
    modifiers['noise'] *= x['noise']
    # Get probability score as sum of modifiers
    p = np.array(list(modifiers.values())).sum()
    # Normalise p to [0, 1]
    p = ((p - minModifier) / (maxModifier - minModifier))
    status = np.random.choice(['DNA', 'Attend'], p=[p, 1-p])
    return pd.Series([p, status])


def generateData(size: int = 50_000, seed: int = 42,
                 noise: float = 0.2) -> pd.DataFrame:
    np.random.seed(seed)
    daysOfWeek = ([
        'Wednesday', 'Tuesday', 'Monday', 'Sunday',
        'Saturday', 'Friday', 'Thursday'
    ])
    logger.info(f'Simulating random dataset with {size} records.')
    data = pd.DataFrame({
        'day': np.random.choice(daysOfWeek, size),
        'priority': np.random.choice(['Urgent', 'Two Week Wait'], size),
        'age': np.random.choice(range(100), size),
        'speciality': np.random.choice(['Ophthalmology', 'Audiology'], size),
        'firstAppointment': np.random.choice([True, False], size),
        'consultationMedia': np.random.choice(['Remote', 'In-Person'], size),
        'site': np.random.choice(['Fairview', 'Lakeside'], size),
        'noise': np.random.uniform(-1, 0, size) # Modifier for attendance weight
    })
    logger.info(f'Setting target status probabilistically.')
    data[['probability', 'status']] = data.apply(
        _setProb, args=(noise,), axis=1)
    data['status'] = (data['status'] == 'DNA').astype(int)
    return data


def writeConfig(config: str = None):
    catCols = ['day', 'priority', 'speciality', 'consultationMedia', 'site']
    boolCols = ['firstAppointment']
    numericCols = ['age']
    config_settings = ({
        'input': 'DNAttend-example.csv',
        'finalModel': 'catboost',
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
