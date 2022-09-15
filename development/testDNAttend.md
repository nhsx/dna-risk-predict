---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%matplotlib inline
from sklearn.base import clone
from scipy.stats import randint, uniform
from dnattend.data import generateData
from dnattend.train import trainModel, splitData, refitAllData
from dnattend.test import getFeatureImportance, plotROC, predict, evaluate
```

```python
df = generateData(size=50_000, seed=42)
```

```python
data = splitData(df, target='status', train_size=0.7, test_size=0.15, val_size=0.15)
```

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

hyperParams = ({
    'estimator__depth':           randint(4, 10),
    'estimator__l2_leaf_reg':     randint(2, 10),
    'estimator__random_strength': uniform.rvs(0, 10, size=100),
})

model, params = trainModel(data, hyperParams=hyperParams, **trainingParams)
```

```python
oldParams = model.get_params()
```

```python
featureImportances = getFeatureImportance(model)
fig = featureImportances.plot.barh()
fig.figure.savefig('../README_files/featureImportances.svg', dpi=300)
```

```python
fig, ax = plotROC(model, data)
fig.figure.savefig('../README_files/ROCcurve.svg', dpi=300)
```

```python
report = evaluate(model, data)
```

```python
model = refitAllData(model, params, data)
```

```python
df[['Attend', 'DNA', 'class']] = predict(model, df)
```

```python

```
