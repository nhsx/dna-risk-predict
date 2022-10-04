#!/usr/bin/env python3

import dnattend.utils as utils
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform
from category_encoders.woe import WOEEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, precision_recall_curve

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


def _buildPreProcessor(catCols, numericCols, boolCols, mode: str = 'catboost'):
    catPipe = SimpleImputer(strategy='constant')
    if mode == 'logistic':
        catPipe =  Pipeline([('impute', catPipe), ('encode', WOEEncoder())])
    transformers = ([
        ('categories', catPipe, catCols),
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
        tuneThresholdBy: str = 'f1',
        seed: int = 42,
        verbose: int = 0,
        nJobs: int = 1):

    np.random.seed(seed)
    for colList in [catCols, numericCols, boolCols]:
        colList = [] if colList is None else colList
        for col in colList.copy():
            if col not in data['X_train']:
                logger.error(f'Feature "{col}" not in data - removing.')
                colList.remove(col)
    if (not catCols) and (not numericCols) and (not boolCols):
        raise ValueError('No features provided.')
    assert tuneThresholdBy in ['f1', 'roc']

    models = {}
    for mode in ['logistic', 'catboost']:
        if mode == 'catboost':
            model, params = _trainCatBoost(
                data, catCols, numericCols,
                boolCols, cvFolds, catboostIterations,
                hypertuneIterations, evalIterations,
                earlyStoppingRounds, hyperParams,
                verbose, nJobs)
        else:
            model, params = _trainLogistic(data, catCols, numericCols, boolCols)
        params = _fixParams(params)
        model = _rebuildPipeline(model, mode)
        _ = model.set_params(**params)
        model = refitData(model, data, noTest=True)

        # Tune decision threshold to balance FPR and TPR
        logger.info('Optimising decision threshold.')
        threshold = _tuneThreshold(
            model, data['X_train'], data['y_train'], tuneThresholdBy)

        logger.info(f'Setting decision threshold to {threshold:.3f}.')
        params['preprocess__prepare__decisionThreshold'] = threshold
        _ = model.set_params(
            preprocess__prepare__decisionThreshold=threshold,
        )
        models[mode] = {'model': model, 'params': params}
    return models


def _trainLogistic(
        data: dict, catCols: list = None,
        numericCols: list = None, boolCols: list = None):

    logger.info('Running base logistic regression model...')
    preProcessor = _buildPreProcessor(
        catCols, numericCols, boolCols, mode='logistic')
    baseEstimator = LogisticRegression(
        class_weight='balanced', random_state=np.random.randint(1e9))
    model = Pipeline(steps=[
        ('preprocess',     preProcessor),
        ('estimator',      baseEstimator),
    ])
    model = model.fit(data['X_train'], data['y_train'])
    params = {}
    return model, params


def _trainCatBoost(
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
        verbose: int = 0,
        nJobs: int = 1):

    if (hyperParams is None) or (not isinstance(hyperParams, dict)):
        hyperParams = ({
            'estimator__depth':           randint(4, 10),
            'estimator__l2_leaf_reg':     randint(2, 10),
            'estimator__random_strength': uniform.rvs(0, 10, size=100),
        })

    logger.info('Building catboost classifier...')
    logger.info('Constructing pre-processing + estimator Pipeline.')
    preProcessor = _buildPreProcessor(
        catCols, numericCols, boolCols, mode='catboost')

    logger.info('Estimating class weights from data.')
    class_weights = _getClassWeights(data['y_train'])

    # Combine processor and modelling steps into a Pipeline object
    catColIdx = preProcessor.named_steps['prepare'].catColIdx
    baseEstimator = CatBoostClassifier(
        cat_features=catColIdx,
        eval_metric='Logloss',
        class_weights=class_weights,
        allow_writing_files=False,
        iterations=catboostIterations, verbose=verbose,
        random_seed=np.random.randint(1e9))
    model = Pipeline(steps=[
        ('preprocess',     preProcessor),
        ('estimator',      baseEstimator),
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

    return model, params


def _tuneThreshold(model, X_train, y_train, mode):
    assert mode in ['f1', 'roc']
    y_trainInt = y_train.apply(lambda x: 1 if x == model.classes_[1] else 0)
    trainPredictProb = model.predict_proba(X_train)[:,1]
    if mode == 'f1':
        logger.info('Tuning threshold by F1-score.')
        precision, recall, thresholds = precision_recall_curve(
            y_trainInt, trainPredictProb)
        fscore = (2 * precision * recall) / (precision + recall)
        idx = np.argmax(fscore)
        optimalThreshold = thresholds[idx]
    else:
        logger.info('Tuning threshold by ROC.')
        fpr, tpr, thresholds = roc_curve(
            y_trainInt, trainPredictProb, drop_intermediate=False)
        idx = np.argmin(np.abs(fpr + tpr - 1))
        optimalThreshold = thresholds[idx]
    return optimalThreshold


def _fixParams(params: dict):
    """ Fix the path to parameters """
    newParams = {}
    for param, val in params.items():
        if param.startswith('estimator__'):
            param = 'estimator__base_estimator' + param[9:]
        newParams[param] = val
    return newParams


def _rebuildPipeline(model, mode: str = 'catboost'):
    """ Rebuild pipeline with  CalibratedClassifierCV """
    catCols = model.get_params()['preprocess__prepare__catCols']
    numericCols = model.get_params()['preprocess__prepare__numericCols']
    boolCols = model.get_params()['preprocess__prepare__boolCols']
    preProcessor = _buildPreProcessor(catCols, numericCols, boolCols, mode)
    catColIdx = preProcessor.named_steps['prepare'].catColIdx
    if mode == 'catboost':
        seed = model.get_params()['estimator__random_seed']
        class_weights = model.get_params()['estimator__class_weights']
        estimator = CatBoostClassifier(
            cat_features=catColIdx, eval_metric='Logloss',
            class_weights=class_weights, verbose=0,
            random_seed=seed, allow_writing_files=False)
    else:
        seed = model.get_params()['estimator__random_state']
        estimator = LogisticRegression(
            class_weight='balanced', verbose=0, random_state=seed)
    model = Pipeline(steps=[
        ('preprocess',     preProcessor),
        ('estimator',      CalibratedClassifierCV(estimator)),
    ])
    return model


def refitData(model, data, noTest=False):
    """ Perform final refit with full data """
    if noTest:
        logger.info('Refitting model with tuned parameters '
                    'on training + validation dataset.')
        X = [data['X_train'], data['X_val']]
        y = [data['y_train'], data['y_val']]
    else:
        logger.info('Refitting model with tuned parameters on full dataset.')
        X = [data['X_train'], data['X_test'], data['X_val']]
        y = [data['y_train'], data['y_test'], data['y_val']]
    logger.info('Recalibrating probabilities with cross-validation.')
    model.fit(pd.concat(X), pd.concat(y))
    return model
