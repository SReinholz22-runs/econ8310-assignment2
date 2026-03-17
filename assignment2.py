# Assignment 3 – Meal Prediction
# Comments added with assistance from ChatGPT

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training and testing datasets
training_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
testing_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Remove non-feature columns from testing data
testing_data = testing_data.drop(['id', 'DateTime', 'meal'], axis=1, errors='ignore')

# Create feature matrix (x) and target variable (y)
x = training_data.drop(['id', 'DateTime', 'meal'], axis=1, errors='ignore')
y = training_data['meal']

# Initialize Random Forest model
model = RandomForestClassifier(
    n_estimators=300,        # number of trees
    n_jobs=-1,               # use all CPU cores
    max_depth=15,            # limit tree depth
    min_samples_leaf=2,      # minimum samples per leaf
    min_samples_split=5,     # minimum samples to split
    random_state=42          # reproducibility
)

# Fit model on training data
modelFit = model.fit(x, y)

# Generate predictions for testing data
pred = modelFit.predict(testing_data)

# Convert predictions to integer series (0 or 1)
pred = pd.Series(pred).astype(int)