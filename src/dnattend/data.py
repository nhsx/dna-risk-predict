#!/usr/bin/env python3

""" Simulate some dummy data for testing """


import numpy as np
import pandas as pd

def _setProb(x: pd.Series) -> pd.Series:
    """ Simulate DNA probability with artificial values """
    modifiers = ({
        'day': 0.05,
        'priority': 0.1,
        'firstAppointment': 0.2,
        'consultationMedia': 0.3,
        'speciality': 0.35,
        'site': 0.4
    })
    maxModifier = np.array(list(modifiers.values())).sum()
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
    # Get probability score as sum of modifiers
    p = np.array(list(modifiers.values())).sum()
    # Normalise p to [0, 1]
    p = ((p - minModifier) / (maxModifier - minModifier))
    if p > 0.65:
        status = 'DNA'
    elif p < 0.35:
        status = 'Attend'
    else:
        status = np.random.choice(['DNA', 'Attend'], p=[p, 1-p])
    return pd.Series([p, status])


def generateData(size: int = 50_000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    daysOfWeek = ([
        'Wednesday', 'Tuesday', 'Monday', 'Sunday',
        'Saturday', 'Friday', 'Thursday'
    ])
    data = pd.DataFrame({
        'day': np.random.choice(daysOfWeek, size),
        'priority': np.random.choice(['Urgent', 'Two Week Wait'], size),
        'age': np.random.choice(range(100), size),
        'speciality': np.random.choice(['Ophthalmology', 'Audiology'], size),
        'firstAppointment': np.random.choice([True, False], size),
        'consultationMedia': np.random.choice(['Remote', 'In-Person'], size),
        'site': np.random.choice(['Fairview', 'Lakeside'], size)
    })
    data[['probability', 'status']] = data.apply(_setProb, axis=1)
    return data
