# DNAttend - ML framework for predicting patient non-attendance

## Train, test and validate a CatBoost Classifier for predicting patient non-attendance (DNA)

## Table of contents

  * [Installation](#installation)
  * [Usage](#usage)

## Installation

```bash
pip install git+https://github.com/nhsx/dna-risk-predict.git
```

## Usage

#### Generate Example Data

```python
from dnattend.data import generateData

# Randomly generate some artificial attendance data
data = generateData(size=50_000, seed=42)
```
