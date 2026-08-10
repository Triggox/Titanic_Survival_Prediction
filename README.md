# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using Logistic Regression.

## Overview

This project uses the classic Titanic dataset to build a binary classification model that predicts whether a passenger survived the Titanic disaster based on features like age, sex, passenger class, fare, and embarkation port.

## Dataset

The dataset (`train.csv`) contains 891 passenger records with the following features:

| Feature | Description |
|---------|-------------|
| PassengerId | Unique passenger ID |
| Survived | Survival (0 = No, 1 = Yes) |
| Pclass | Ticket class (1, 2, 3) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C, Q, S) |

## Approach

1. **Data Exploration** — Initial inspection of data shape, types, summary statistics, and null values
2. **Data Cleaning** — Handling missing values:
   - Age: filled with median
   - Embarked: filled with mode
   - Cabin: dropped (too many missing values)
3. **Feature Engineering** — Dropped non-predictive columns (Name, Ticket, PassengerId), label-encoded Sex, and one-hot-encoded Embarked
4. **Train-Test Split** — 80/20 split with `random_state=42`
5. **Feature Scaling** — StandardScaler applied to normalize feature ranges
6. **Model** — Logistic Regression (default parameters)
7. **Evaluation** — Accuracy, confusion matrix, precision, recall, and F1-score

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **81.01%** |
| Precision (Survived) | 0.79 |
| Recall (Survived) | 0.74 |
| F1-score (Survived) | 0.76 |

## Usage

```bash
# Activate virtual environment
source ../venv/bin/activate

# Launch Jupyter
jupyter notebook Titanic_Prediction.ipynb
```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
