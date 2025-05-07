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

In this project, I used a customer dataset and applied classification models to predict churn, including:

- Neural Network
- Random Forest

## Basic NN

<pre><code>
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Let's only see France
data_France = data_churn[data_churn['Geography'] == 'France']

# Features and target
X = data_France[['Age']].values
y = data_France['Exited'].values

# Normalize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1015)

import matplotlib.pyplot as plt
import tensorflow as tf

# Our Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_split=0.1)
</code></pre>

## Results
The models were evaluated using metrics like:

- Accuracy
- Precision & Recall
- ROC AUC Score


## Discussion

## Conclusion

The final model achieved over 83% accuracy and highlighted key churn factors such as age, account activity, and credit card usage.

