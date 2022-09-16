# DNAttend - ML framework for predicting patient non-attendance

## Train, test and validate a CatBoost Classifier for predicting patient non-attendance (DNA)

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
![build: status](https://github.com/nhsx/dna-risk-predict/actions/workflows/tests.yaml/badge.svg?event=push)




## Table of contents

  * [Installation](#installation)
  * [Workflow](#workflow)
  * [Usage](#usage)
    * [Generate Example Data](#generate-example-data)
    * [Split Data (Test, Train, Validation)](#split-data-test-train-validation)
    * [Train Model](#train-model)
    * [Evaluate Model](#evaluate-model)
      * [Feature Importance](#feature-importance)
      * [ROC Curve](#roc-curve)
      * [Evaluation Report](#evaluation-report)
    * [Re-fit model with all data](#re-fit-model-with-all-data)
  * [Generate Predictions](#generate-predictions)
  * [Contributing](#contributing)
  * [License](#license)
  * [Contact](#contact)


## Installation

```bash
pip install git+https://github.com/nhsx/dna-risk-predict.git
```

## Worklow

![workflow](./README_files/DNApredictFlowchart.svg)
 <br> *Overview of DNAttend workflow*

## Usage

```python
from scipy.stats import randint, uniform
from dnattend.data import generateData
from dnattend.train import trainModel, splitData, refitAllData
from dnattend.test import getFeatureImportance, plotROC, predict, evaluate
```

### Generate Example Data

```python
# Randomly generate some artificial attendance data
df = generateData(size=50_000, seed=42)
```

### Split Data (Test, Train, Validation)

```python
data = splitData(df, target='status', train_size=0.7, test_size=0.15, val_size=0.15)
```

### Train Model

```python
catCols = ['day', 'priority', 'speciality', 'consultationMedia', 'site']
boolCols = ['firstAppointment']
numericCols = ['age']

trainingParams = ({
    'catCols':             catCols,
    'boolCols':            boolCols,
    'numericCols':         numericCols,
    'cvFolds':             5,
    'catboostIterations':  100,
    'hypertuneIterations': 5,
    'evalIterations':      10_000,
    'earlyStoppingRounds': 10,
    'seed':                42
})
```

```python
# Optional - define estimator hyper-parameter search space
hyperParams = ({
    'estimator__depth':           randint(4, 10),
    'estimator__l2_leaf_reg':     randint(2, 10),
    'estimator__random_strength': uniform.rvs(0, 10, size=100),
})
```

```python
model, params = trainModel(data, hyperParams=hyperParams, **trainingParams)
```

![model](./README_files/modelWorkflow.svg)
 <br> *Summary of scikit-learn Pipeline*

### Evaluate Model

#### Feature Importance

```python
featureImportances = getFeatureImportance(model)
fig = featureImportances.plot.barh()
fig.figure.savefig('featureImportances.png')
```

![featureImporance](./README_files/featureImportances.svg)


#### ROC Curve

```python
fig, ax = plotROC(model, data)
fig.figure.savefig('ROCcurve.svg')
```


![ROC](./README_files/ROCcurve.svg)

#### Evaluation Report
The `evaluate()` function computes a comprehensive set of performance metrics using the `test` data.

```python
report = evaluate(model, data)

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

### Re-fit model with all data
Following parameterisation and validation the `refitAllData()` function can be used to refit a new model on the whole data set.

```python
model = refitAllData(model, params, data)
```

### Generate Predictions
The trained model is now ready to be used.
Predictions should be made with the `predict()` wrapper function - this ensures the tuned decision threshold is correct applied when assigning classes.
The output of `predict()` includes the decision class (i.e.`Attend` and `DNA`) and the underlying probabilities of theses classes.
The output results of this example can be found [here](./README_files/example-data-predictions.csv)

```python
df[['Attend', 'DNA', 'class']] = predict(model, df)
```

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### License

Distributed under the MIT License. _See [LICENSE](./LICENSE) for more information._

### Contact

If you have any other questions please contact the author **[Stephen Richer](https://www.linkedin.com/in/stephenricher/)**
at stephen.richer@proton.me
