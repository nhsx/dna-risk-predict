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
    config_settings = (''
        'input: DNAttend-example.csv    # Path to input data (Mandatory).\n'
        'target: status                 # Column name of target (Mandatory).\n'
        'DNAclass: 1                    # Value of target corresponding to DNA.\n'
        'out: .                         # Output directory to save results.\n'
        'finalModel: catboost           # Method to train final model (catboost or logistic).\n'
        'catCols:                       # Column names of categorical features.\n'
        '    - day\n'
        '    - priority\n'
        '    - speciality\n'
        '    - consultationMedia\n'
        '    - site\n'
        'boolCols:                      # Column names of boolean features.\n'
        '    - firstAppointment\n'
        'numericCols:                   # Column names of numeric features.\n'
        '    - age\n'
        'train_size: 0.7                # Proportion of data for training.\n'
        'test_size: 0.15                # Proportion of data for testing.\n'
        'val_size: 0.15                 # Proportion of data for validation.\n'
        'tuneThresholdBy: f1            # Metric to tune decision threshold (f1 or roc).\n'
        'cvFolds: 5                     # Hyper-tuning cross-validations.\n'
        'catboostIterations: 100        # Hyper-tuning CatBoost iterations.\n'
        'hypertuneIterations: 5         # Hyper-tuning parameter samples.\n'
        'evalIterations: 10_000         # Upper-limit over-fit iterations.\n'
        'earlyStoppingRounds: 10        # Over-fit detection early stopping rounds.\n'
        'encoding: latin-1              # Encoding to use for reading files.\n'
        'seed: 42                       # Seed to ensure workflow reproducibility.\n'
    )
    if config is None:
        sys.stderr.write(config_settings)
    else:
        with open(config, 'w') as fh:
            fh.write(config_settings)
