---
layout: md_layout
title: Bank Churn
---

# Bank Churn Prediction Project

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
## Creating a train/test dataset

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
```

## Training a XGBoost Model

## Tuning Random Forest Hyperparameters

## Tuning XGBoost Hyperparameters

## Results

  
## Discussion

## Conclusion


