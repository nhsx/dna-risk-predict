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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dnattend.data import generateData
from dnattend.train import trainModel, splitData
```

```python
data = generateData(size=100_000, seed=42)
X = data.copy()
y = X.pop('status')
```

```python
X_train, X_test, X_val, y_train, y_test, y_val = (
    splitData(X, y, train_size=0.7, test_size=0.15, val_size=0.15)
)
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
model, params = trainModel(X_train, y_train, X_val, y_val, **trainingParams)
```

```python
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, RocCurveDisplay
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
y_trainInt = y_train.apply(lambda x: 1 if x == model.classes_[1] else 0)
y_predPos = model.predict_proba(X_train)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_trainInt, y_predPos, drop_intermediate=False)
AUC = roc_auc_score(y_trainInt, y_predPos)

idx = np.argmin(np.abs(fpr + tpr - 1))
optimalThreshold = thresholds[idx]
```

```python
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axhline(tpr[idx], xmax=fpr[idx], ls='--', alpha=0.5, c='black')
ax.axvline(fpr[idx], ymax=tpr[idx], ls='--', alpha=0.5, c='black')
ax.scatter(fpr[idx], tpr[idx], c='black')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
label = f'AUC = {AUC:.2f}, Optimal Threshold = {optimalThreshold:.2f}'
ax.legend(labels=[label], loc='lower right')
fig.show()
```

```python
importances = pd.Series(
    model.named_steps['estimator'].feature_importances_,
    model.named_steps['preprocess'].named_steps['prepare'].validCols
).sort_values(ascending=False)
importances
```

```python
predictions = predict(model, X_test, threshold=optimalThreshold)
report = classification_report(y_test, predictions['class'], output_dict=True)
```

```python
report
```

```python
data[['Attend', 'DNA', 'class']] = predict(model, data, threshold=optimalThreshold)
```

```python
data
```

```python

```
