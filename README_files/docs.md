# DNAttend - ML framework for predicting patient non-attendance

## Additional documentation and example visualisations

**Note: This documentation is still a work progress and will be updated.**

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
![build: status](https://github.com/nhsx/dna-risk-predict/actions/workflows/tests.yaml/badge.svg)

## Table of contents

  * [Workflow](#workflow)
  * [Model Evaluation](#model-evaluation)
      * [Feature Importance](#feature-importance)
      * [ROC Curve](#roc-curve)
      * [Precision-Recall Curve](#precision-recall-curve)
      * [Calibration Curve](#calibration-curve)
      * [Evaluation Report](#evaluation-report)


## Worklow
DNAttend trains two models independently; a baseline logistic regression model and a CatBoost model.
The CatBoost model is trained via a cross-validated randomised hyper-parameter search with over-fit detection.
In addition, over-fit detection is performed using a holdout validation set to determine the optimal boosting iterations.
Output probability of both models are calibrated via cross-validation.
Finally, decision thresholds are tuned, using the training dataset, to optimise either the ROC or F1 score.
This choice of threshold metric is determined by the `tuneThresholdBy` option of the configuration file (defult = f1).

![workflow](./DNApredictFlowchart.png)
 <br> *Detailed overview of the DNAttend workflow*

### Model Evaluation

#### Feature Importance

```python
featureImportances = test.getFeatureImportance(models['catboost']['model'])
fig = featureImportances.plot.barh()
fig.figure.savefig('featureImportances.png')
```

![featureImportance](./featureImportances.pdf)
 <br> *Feature Importances.*

#### ROC Curve

```python
fig, ax = test.plotROC(models, data)
fig.figure.savefig('ROCcurve.png')
```

![ROC](./ROCcurve.png)
 <br> *Receiver Operating Characteristic curve for both CatBoost and Logistic Model.*

#### Precision-Recall Curve

```python
fig, ax = test.plotPrecisionRecall(models, data)
fig.figure.savefig('PRcurve.png', dpi=300)
```

![ROC](./PRcurve.png)
 <br> *Precision-Recall curve for both CatBoost and Logistic Model.*

#### Calibration Curve

```python
fig, ax = test.plotCalibrationCurve(models, data, strategy='quantile')
fig.figure.savefig('CalibrationCurve.png')
```

![ROC](./CalibrationCurve.png)
 <br> *Calibration curve for both CatBoost and Logistic Model.*

#### Evaluation Report
The `evaluate()` function computes a comprehensive set of performance metrics using the `test` data.

```python
report = test.evaluate(models['catboost']['model'], data)

print(report)
{
    'Attend': {
        'precision': 0.7976354138025845,
        'recall':    0.7815193965517241,
        'f1-score':  0.7894951694108042,
        'support':   3712
    },
    'DNA': {
        'precision': 0.7901138716356108,
        'recall':    0.8057534969648984,
        'f1-score':  0.7978570495230629,
        'support':   3789
    },
    'accuracy':      0.7937608318890814,
    'macro avg': {
        'precision': 0.7938746427190977,
        'recall':    0.7936364467583112,
        'f1-score':  0.7936761094669336,
        'support':   7501
    },
    'weighted avg': {
        'precision': 0.7938360372833652,
        'recall':    0.7937608318890814,
        'f1-score':  0.7937190280623637,
        'support':   7501
    }
}

```
