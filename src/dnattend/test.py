#!/usr/bin/env python3

import dnattend.utils as utils
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay


logger = logging.getLogger(__name__)


def getFeatureImportance(model):
    calClassifiers = model.named_steps['estimator'].calibrated_classifiers_
    importances = 0
    for classifier in calClassifiers:
        importances += classifier.base_estimator.feature_importances_
    importances /= len(calClassifiers)
    # model.named_steps['estimator'].base_estimator.feature_importances_
    importances = pd.Series(
        importances,
        model.named_steps['preprocess'].named_steps['prepare'].validCols
    ).sort_values(ascending=True)
    return importances


def _processTestData(model, data):
    y_test = data['y_test'].apply(
        lambda x: 1 if x == model.classes_[1] else 0)
    test_pred_proba = model.predict_proba(data['X_test'])[:,1]
    return y_test, test_pred_proba


def plotROC(model, data, figsize=None):
    y_test, test_pred_proba = _processTestData(model, data)
    AUC = roc_auc_score(y_test, test_pred_proba)
    fpr, tpr, thresholds = roc_curve(
        y_test, test_pred_proba, drop_intermediate=False)
    idx = np.argmin(np.abs(fpr + tpr - 1))
    posClass = model.classes_[1]
    threshold = model.get_params()['preprocess__prepare__decisionThreshold']

    fig, ax = plt.subplots(figsize=figsize)
    RocCurveDisplay.from_estimator(
        model, data['X_test'], data['y_test'], ax=ax)
    ax.axline((0, 0), slope=1, ls='--', color='red')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axhline(tpr[idx], xmax=fpr[idx], ls='--', alpha=0.5, c='black')
    ax.axvline(fpr[idx], ymax=tpr[idx], ls='--', alpha=0.5, c='black')
    ax.scatter(fpr[idx], tpr[idx], c='black')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    label = f'AUC = {AUC:.2f}, {posClass} Threshold = {threshold:.3f}'
    ax.legend(labels=[label], loc='lower right')
    return fig, ax


def plotCalibrationCurve(model, data, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    CalibrationDisplay.from_estimator(
        model, data['X_test'], data['y_test'], ref_line=False, ax=ax)
    ax.axline((0, 0), slope=1, ls='--', color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(labels=['DNAttend Model'], loc='lower right')
    return fig, ax


def predict(model, X):
    """ Generate predictions using trained model """
    try:
        param = 'preprocess__prepare__decisionThreshold'
        threshold = model.get_params()[param]
    except KeyError:
        threshold = 0.5
    classes = model.classes_
    out = pd.DataFrame(model.predict_proba(X), columns=classes)
    out['class'] = out[classes[1]].apply(
        lambda x: classes[0] if x < threshold else classes[1])
    return out.values


def evaluate(model, data):
    """ Generate classification report using test data """
    predictions = predict(model, data['X_test'])[:,2]
    report = classification_report(
        data['y_test'], predictions['class'], output_dict=True)
    return report
