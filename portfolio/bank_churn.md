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
- 

## Machine Learning Model

```python
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

X = data_churn_1.drop('Exited', axis=1)
y = data_churn_1['Exited']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=1015)
```

## Results
The models were evaluated using metrics like:

- Accuracy
- Confusion Matrix – to analyze the number of correct and incorrect predictions for each class
- ROC AUC Score – to measure the model's ability to distinguish between classes
  
## Discussion

## Conclusion

The final model achieved over 83% accuracy and highlighted key churn factors such as age, account activity, and credit card usage.

