# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:03:30 2024

@author: 91966
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Initialize KNN classifier with a chosen k (number of neighbors)
k = 5  # You can adjust the value of k to improve performance
model = KNeighborsClassifier(n_neighbors=k)

# Train the model
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)

# Calculate and print accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"The model correctly classifies approximately   {accuracy * 100:.2f}% of the test samples based on the given dataset.")
