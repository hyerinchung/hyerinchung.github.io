---
layout: md_layout
title: Bank Churn
---

# Bank Churn Prediction using Machine Learning Algorithms

![Bank](https://github.com/hyerinchung/hyerinchung.github.io/blob/main/images/bank_head.jpg?raw=true)


## Introduction

Customer churn refers to losing customers or clients over a given period, which is typically expressed as a percentage.
Since acquiring new customers is typically more expensive than retaining existing ones, predicting customer churn is crucial for many companies.
A high churn rate indicates that a company is losing a significant number of subscribers, which can slow growth and negatively affect sales and profitability.
Then, how does the prediction work?
I will begin by performing Exploratory Data Analysis on the dataset and applying various machine learning classification algorithms to identify the most effective model for predicting bank customer churn.


## Dataset and Features
This project uses a dataset of 10,000 bank customers, with their features. The data was sourced from Kaggle.
It includes the following features:

- RowNumber: The record number, which doesn't affect the outcome.
- CustomerId: A unique identifier for the customer, with no impact on churn.
- Surname: The customer’s last name, irrelevant to leaving the bank.
- CreditScore: The customer’s credit score; higher scores reduce churn risk.
- Geography: The customer’s location, which may influence churn.
- Gender: The customer’s gender, potentially affecting their decision to leave.
- Age: The customer’s age; older customers tend to stay longer.
- Tenure: The number of years with the bank; longer tenure reduces churn.
- Balance: The customer’s account balance; higher balances reduce churn.
- NumOfProducts: The number of bank products the customer holds.
- HasCrCard: Whether the customer has a credit card; cardholders are less likely to leave. (0=No, 1=Yes)
- IsActiveMember: Whether the customer is active; active members are less likely to leave. (0=No, 1=Yes)
- EstimatedSalary: The customer’s estimated salary; lower salaries increase churn risk.
- Exited: Whether the customer left the bank. (0=No, 1=Yes)

'Exited' variable is the target variable that we need to predict.

## Methodology

In this project, I used a customer dataset and applied machine learning models to predict churn, including:

- Neural Network
- Random Forest
- XGBoost

The models were evaluated and compared using:

- Accuracy
- ROC AUC Score

## Data Processing

```python
# Load data
df = pd.read_csv("/content/Churn_Modelling.csv")
print(df.head())

# Generate credit_index
df['credit_index'] = (df['CreditScore'] // 100) * 100

# Drop unnecessary columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# Drop NA
df = df.dropna()

# Encoding categorical variable

le_gender = LabelEncoder()
le_geo = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Geography"] = le_geo.fit_transform(df["Geography"])
```
## Creating a Train/test Dataset

```python
# Split Feature and Target

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Split test and train set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1015, stratify=y)
```

## Training a Random Forest Model

```python
# Random Forest Model

rf = RandomForestClassifier(n_estimators=100, random_state=1015)
rf.fit(X_train, y_train)

# Prediction and Evaluation

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

```

## Training a XGBoost Model

```python
# XGBoost model

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1015,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Prediction

y_pred_xgb = xgb_model.predict(X_test)

```

## Tuning Random Forest Hyperparameters
```python
# Basic Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameters
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Randomized Search
random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=1015,
    scoring='roc_auc'
)

# Training
random_search_rf.fit(X_train, y_train)

# Prediction
best_rf = random_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
```

## Tuning XGBoost Hyperparameters
```python
# Basic XGBoost
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=1015
)

# Parameters
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,                # 30 combinations
    scoring='roc_auc',        # AUC Evaluation
    cv=3,                     # 3-fold cross validation
    verbose=1,
    n_jobs=-1,
    random_state=1015
)

# Training
random_search.fit(X_train, y_train)

# Testing best model
best_xgb = random_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:, 1]
```

## Results

  
## Discussion

## Conclusion


