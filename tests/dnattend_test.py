#!/usr/bin/env python3

import pytest
from dnattend import main


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_template():
    main.simulate_cli(out='DNAttend-example.csv', config='config.yaml')
    main.train_cli('config.yaml')
    main.retrain_cli('config.yaml')
    main.predict_cli(
        'DNAttend-example.csv', 'catboost-final.pkl', out=None, verify=True)
