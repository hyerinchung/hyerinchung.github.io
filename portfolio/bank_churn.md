---
layout: default
title: Bank Churn Prediction
---
<style>
  h1 {
    border: none;  /* h1 태그에 적용된 border 제거 */
    margin: 0;     /* margin 제거 */
    padding: 0;    /* padding 제거 */
  }

  hr {
    display: none;  /* 수평선 제거 */
  }

  /* 추가로 전체적인 스타일 조정 */
  body {
    margin: 0;
    padding: 0;
  }

  .content {
    margin-top: 20px; /* 내용 위쪽 여백을 조정 */
  }
</style>

# 🏦 Bank Churn Prediction Project

Welcome to my Bank Churn Prediction project! This project focuses on identifying customers who are likely to leave a bank using machine learning techniques.

## 📌 Overview

In this project, I used a customer dataset and applied classification models to predict churn, including:

- Neural Network
- Random Forest
- XGBoost

## 📊 Features Used

Key features in the dataset:

- Credit Score
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## 🧪 Basic NN

```python
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

## 🧠 Model Performance

The models were evaluated using metrics like:

- Accuracy
- Precision & Recall
- ROC AUC Score

## 📎 Conclusion

The final model achieved over 83% accuracy and highlighted key churn factors such as age, account activity, and credit card usage.

[🔙 Back to Home](../index.html)
