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

- Age
- Country
- Gender
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## Methodology

In this project, I used a customer dataset and applied machine learning models to predict churn, including:

- XGBoost
- Random Forest

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

## Results
The models were evaluated using metrics like:

- Accuracy
- Confusion Matrix – to analyze the number of correct and incorrect predictions for each class
- ROC AUC Score – to measure the model's ability to distinguish between classes
  
## Discussion

## Conclusion

The final model achieved over 83% accuracy and highlighted key churn factors such as age, account activity, and credit card usage.

