# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:59:27 2024

@author: 91966
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Define input (X) and output (Y)
X = data[features]
Y = data[target]
# Plot all parameters on one graph with adjusted scaling
plt.figure(figsize=(12, 8))
for feature in features:
    plt.scatter(X[feature], Y, label=feature, alpha=0.6, s=15)  # Adjust marker size and transparency


plt.title("Input Features vs GPA", fontsize=16)
plt.xlabel("Feature Value", fontsize=14)
plt.ylabel("GPA", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
































