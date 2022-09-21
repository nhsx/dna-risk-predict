#!/usr/bin/env python3

import dnattend.utils as utils
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold


logger = logging.getLogger(__name__)


def splitData(
        data, target, train_size: float = 0.8, test_size: float = 0.1,
        val_size: float = 0.1, seed: int = None):
    """ Split data into test / train / validation """

    assert train_size + test_size + val_size == 1
    logger.info(f'Splitting data: train ({train_size:.1%}) : '
                f'test ({test_size:.1%}) : validation ({val_size:.1%}).')
    rng = np.random.default_rng(seed)

    X = data.copy()
    y = X.pop(target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_size),
        random_state=rng.integers(1e9))

    split2Size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=split2Size,
        random_state=rng.integers(1e9))

    return ({
        'X_train': X_train.copy(), 'y_train': y_train.copy(),
        'X_test': X_test.copy(), 'y_test': y_test.copy(),
        'X_val': X_val.copy(), 'y_val': y_val.copy()
    })


def _getClassWeights(y):
    # Set class weight to balanced probalities for more easy interpretation
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y)
    return  dict(zip(classes, weights))


def _buildPreProcessor(catCols, numericCols, boolCols):
    transformers = ([
        ('categories', SimpleImputer(strategy='constant'), catCols),
        ('numeric',    SimpleImputer(strategy='mean'), numericCols),
        ('boolean',    SimpleImputer(strategy='most_frequent'), boolCols),
    ])
    featureTransformer = ColumnTransformer(
        transformers=transformers, remainder='drop')

    preProcessor = Pipeline(steps=[
        ('prepare',         utils._prepareData(catCols, numericCols, boolCols)),
        ('columnTransform', featureTransformer),
    ])
    return preProcessor


def trainModel(
        data: dict,
        catCols: list = None,
        numericCols: list = None,
        boolCols: list = None,
        cvFolds: int = 5,
        catboostIterations: int = 100,
        hypertuneIterations: int = 5,
        evalIterations: int = 10000,
        earlyStoppingRounds: int = 10,
        hyperParams: dict = None,
        seed: int = 42,
        verbose: int = 0,
        nJobs: int = 1):

    np.random.seed(seed)

    if (hyperParams is None) or (not isinstance(hyperParams, dict)):
        hyperParams = ({
            'estimator__depth':           randint(4, 10),
            'estimator__l2_leaf_reg':     randint(2, 10),
            'estimator__random_strength': uniform.rvs(0, 10, size=100),
        })

    if (catCols is None) and (numericCols is None) and (boolCols is None):
        raise ValueError('No features provided.')

    logger.info('Estimating class weights from data.')
    class_weights = _getClassWeights(data['y_train'])

    logger.info('Constructing pre-processing + estimator Pipeline.')
    preProcessor = _buildPreProcessor(catCols, numericCols, boolCols)
    # Combine processor and modelling steps into a Pipeline object
    catColIdx = preProcessor.named_steps['prepare'].catColIdx
    model = Pipeline(steps=[
        ('preprocess',     preProcessor),
        ('estimator',      CatBoostClassifier(
            cat_features=catColIdx,
            eval_metric='Logloss',
            class_weights=class_weights,
            allow_writing_files=False,
            iterations=catboostIterations, verbose=verbose,
            random_seed=np.random.randint(1e9))),
    ])

    logger.info(f'Performing {cvFolds}-fold cross-validated random search '
                 f'of hyper-parameters ({hypertuneIterations} iterations).')
    gridSearch = RandomizedSearchCV(
        model, hyperParams, scoring='neg_log_loss',
        random_state=np.random.randint(1e9), cv=cvFolds,
        refit=False, n_jobs=nJobs, n_iter=hypertuneIterations, verbose=verbose)
    _ = gridSearch.fit(data['X_train'], data['y_train'])

    # Extract best parameters from cross-validated randomised search
    params = gridSearch.best_params_
    params['estimator__iterations'] = evalIterations
    _ = model.set_params(**params)

    # Pre-process the validation set with the tuned model parameters.
    # Required since eval_set is other not processed before CatBoost
    fitPreProcessor = model.named_steps['preprocess'].fit(
        data['X_train'], data['y_train'])
    X_val = fitPreProcessor.transform(data['X_val'])
    evalSet = Pool(X_val, data['y_val'], cat_features=catColIdx)
    logger.info('Re-fitting tuned model to estimate '
                 'optimal iterations using Early Stopping.')
    _ = model.fit(
        data['X_train'], data['y_train'], estimator__eval_set=evalSet,
        estimator__early_stopping_rounds=earlyStoppingRounds)

    # Update iteration parameter to optimal and write to file
    bestIteration = model.named_steps['estimator'].get_best_iteration()
    logger.info(f'Setting iterations to {bestIteration}.')
    params['estimator__iterations'] = bestIteration

    # Tune decision threshold to balance FPR and TPR
    logger.info('Optimising decision threshold to balance '
                'true-negative and true-positive rates.')
    threshold = utils._tuneThreshold(model, data['X_train'], data['y_train'])
    logger.info(f'Setting decision threshold to {threshold:.3f}.')
    params['preprocess__prepare__decisionThreshold'] = threshold
    _ = model.set_params(
        preprocess__prepare__decisionThreshold=threshold,
    )

    return model, params


def _rebuildPipeline(model):
    """ Rebuild unbuilt pipeline """
    seed = model.get_params()['estimator__random_seed']
    class_weights = model.get_params()['estimator__class_weights']
    catCols = model.get_params()['preprocess__prepare__catCols']
    numericCols = model.get_params()['preprocess__prepare__numericCols']
    boolCols = model.get_params()['preprocess__prepare__boolCols']
    preProcessor = _buildPreProcessor(catCols, numericCols, boolCols)
    catColIdx = preProcessor.named_steps['prepare'].catColIdx
    model = Pipeline(steps=[
        ('preprocess',     preProcessor),
        ('estimator',      CatBoostClassifier(
            cat_features=catColIdx, eval_metric='Logloss',
            class_weights=class_weights, verbose=0,
            random_seed=seed, allow_writing_files=False)),
    ])
    return model


def refitAllData(model, params, data):
    """ Perform final refit with full data """
    model = _rebuildPipeline(model)
    _ = model.set_params(**params)
    logger.info('Re-fitting model with tuned paramters on full dataset.')
    model.fit(
        pd.concat([data['X_train'], data['X_test'], data['X_val']]),
        pd.concat([data['y_train'], data['y_test'], data['y_val']])
    )
    return model
