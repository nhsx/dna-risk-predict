#!/usr/bin/env python3


import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay


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


def plotROC(models, data, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    labels = []
    for name in ['logistic', 'catboost']:
        model = models[name]['model']
        y_test, test_pred_proba = _processTestData(model, data)
        AUC = roc_auc_score(y_test, test_pred_proba)
        fpr, tpr, thresholds = roc_curve(
            y_test, test_pred_proba, drop_intermediate=False)
        idx = np.nanargmin(np.abs(fpr + tpr - 1))
        RocCurveDisplay.from_estimator(
            model, data['X_test'], data['y_test'], ax=ax)
        if name == 'catboost':
            ax.axhline(tpr[idx], xmax=fpr[idx], ls='--', alpha=0.5, c='black')
            ax.axvline(fpr[idx], ymax=tpr[idx], ls='--', alpha=0.5, c='black')
            ax.scatter(fpr[idx], tpr[idx], c='black')
        label = f'{name}: AUC = {AUC:.2f}, Threshold = {thresholds[idx]:.3f}'
        labels.append(label)
    ax.axline((0, 0), slope=1, ls='--', color='red')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(labels=labels, loc='lower right')
    return fig, ax


def plotPrecisionRecall(models, data, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    labels = []
    for name in ['logistic', 'catboost']:
        model = models[name]['model']
        y_test, test_pred_proba = _processTestData(model, data)
        precision, recall, thresholds = precision_recall_curve(
            y_test, test_pred_proba, pos_label=1)
        fscore = (2 * precision * recall) / (precision + recall)
        idx = np.nanargmax(fscore)
        PrecisionRecallDisplay.from_estimator(
            model, data['X_test'], data['y_test'], ax=ax)
        if name == 'catboost':
            ax.axhline(precision[idx], xmax=recall[idx], ls='--', alpha=0.5, c='black')
            ax.axvline(recall[idx], ymax=precision[idx], ls='--', alpha=0.5, c='black')
            ax.scatter(recall[idx], precision[idx], c='black')
        label = f'{name}: F-score = {fscore[idx]:.2f}, Threshold = {thresholds[idx]:.3f}'
        labels.append(label)
    noSkill = data['y_test'].sum() / len(data['y_test'])
    ax.axline((0, noSkill), (1, noSkill), ls='--', color='red')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(labels=labels, loc='upper right')
    return fig, ax


def plotCalibrationCurve(models, data, figsize=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    names = ['logistic', 'catboost']
    for name in names:
        model = models[name]['model']
        CalibrationDisplay.from_estimator(
            model, data['X_test'], data['y_test'],
            ref_line=False, ax=ax, **kwargs)
    ax.axline((0, 0), slope=1, ls='--', color='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(labels=names, loc='lower right')
    return fig, ax


def predict(model, X):
    """ Generate predictions using trained model """
    try:
        param = 'preprocess__prepare__decisionThreshold'
        threshold = model.get_params()[param]
    except KeyError:
        logger.error('No threshold set - setting to 0.5.')
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
        data['y_test'], predictions, output_dict=True)
    return report
