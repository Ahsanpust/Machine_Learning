Predicting Type 2 Diabetes Using Pima Indian Dataset

Introduction

The Pima Indian Diabetes dataset is a widely used dataset in machine learning for predicting Type 2 diabetes. This document provides a step-by-step guide to predict diabetes using this dataset.

Steps to Solve the Problem

1. Load Necessary Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

2. Load the Dataset

Download the dataset from Kaggle.

# Load the dataset
data_path = "diabetes.csv"  # Update with your file path
data = pd.read_csv(data_path)

# Display the first few rows
print(data.head())

3. Dataset Overview
#Pregnancies: Number of times pregnant.
#Glucose: Plasma glucose concentration.
#BloodPressure: Diastolic blood pressure (mm Hg).
#SkinThickness: Triceps skinfold thickness (mm).
#Insulin: 2-Hour serum insulin (mu U/ml).
#BMI: Body mass index.
#DiabetesPedigreeFunction: Diabetes pedigree function (likelihood based on family history).
#Age: Age (years).
#Outcome: Class label (0 = No diabetes, 1 = Diabetes).

Check for missing values or data quality issues:

print(data.info())
print(data.describe())

4. Handle Missing or Zero Values

Some features might have zero values that are unrealistic (e.g., zero BMI or glucose). Replace them with the median or mean:

# Replace zero values with NaN
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

# Replace NaN with median values
for col in columns_with_zeros:
    data[col].fillna(data[col].median(), inplace=True)

5. Split the Data

Divide the dataset into training and testing sets:

# Features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

6. Feature Scaling

Standardize the features for better performance with models like Logistic Regression:

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

7. Train a Logistic Regression Model

# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

8. Evaluate the Model

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

9. Interpretation of Results

Accuracy: Measures overall correctness.

Confusion Matrix: Provides true positives, true negatives, false positives, and false negatives.

Classification Report: Includes precision, recall, F1-score for both classes.





