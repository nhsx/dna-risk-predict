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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dnattend.data import generateData
from dnattend.train import trainModel, splitData
from dnattend.test import getFeatureImportances, plotRocCurve
```

```python
df = generateData(size=10_000, seed=42)
```

```python
data = splitData(df, target='status', train_size=0.7, test_size=0.15, val_size=0.15)
```

```python
trainingParams = ({
    'catCols': ['day', 'priority', 'speciality', 'consultationMedia', 'site'],
    'boolCols': ['firstAppointment'],
    'numericCols': ['age'],
    'cvFolds': 5,
    'catboostIterations': 100,
    'hypertuneIterations': 5,
    'evalIterations': 10_000,
    'earlyStoppingRounds': 10,
    'seed': 42 
})
```

```python
model, params = trainModel(data, **trainingParams)
```

```python
featureImportances = getFeatureImportances(model)
featureImportances.plot.barh()
```

```python
fig, ax = plotRocCurve(model, data)
```

```python
from sklearn.metrics import classification_report
```

```python
def predict(model, X, threshold=0.5):
    classes = model.classes_
    out = pd.DataFrame(model.predict_proba(X), columns=classes)
    out['class'] = out[classes[1]].apply(
        lambda x: classes[0] if x < threshold else classes[1])
    return out
```

```python
predictions = predict(model, X_test, threshold=optimalThreshold)
report = classification_report(y_test, predictions['class'], output_dict=True)
```

```python
report
```

```python
data[['Attend'), ('probability', 'DNA'), 'class']] = predict(model, data, threshold=optimalThreshold)
```

```python
data
```

```python

```
