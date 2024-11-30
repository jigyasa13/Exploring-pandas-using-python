# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:43:13 2024

@author: 91966
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('student_lifestyle_dataset.csv', sep=',')

# Define input features and output target
features = [
    "Physical_Activity_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Study_Hours_Per_Day"
]
target = "GPA"

# Convert to numeric and drop missing values
data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')
data = data.dropna(subset=features + [target])

# Convert GPA into categories (Binary: Above/Below Average GPA)
data['GPA_Category'] = (data[target] >= data[target].median()).astype(int)  # 1 for Above Average, 0 for Below Average

# Define input (X) and output (Y)
X = data[features]
Y = data['GPA_Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)

# Calculate and print accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

