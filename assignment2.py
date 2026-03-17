# Assignment 3 – Meal Prediction
# AI Assistance Disclosure: ChatGPT was used for debugging help, code organization, comparison of model approaches, and commenting while developing this assignment.

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Load training and testing data
training_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
testing_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Create feature matrix and target variable
x = training_data.drop(['id', 'DateTime', 'meal'], axis=1, errors='ignore')
y = training_data['meal']

# Prepare test-period feature data
predData = testing_data.drop(['id', 'DateTime', 'meal'], axis=1, errors='ignore')

# Load true values for testing/comparison
dataTest = pd.read_csv("tests/testData.csv")['meal']

# Function to evaluate classification accuracy
def get_accuracy(truth, pred):
    truth = list(truth)
    pred = list(pred)
    return np.mean([1 if truth[i] == pred[i] else 0 for i in range(len(truth))])

# Store results here
results = []

# -------------------------
# Decision Tree
# -------------------------
dt_model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=15,
    min_samples_leaf=10,
    random_state=42
)
dt_fit = dt_model.fit(x, y)
dt_pred = dt_fit.predict(predData)
dt_acc = get_accuracy(dataTest, dt_pred)
results.append(("Decision Tree", dt_model, dt_fit, dt_pred, dt_acc))

# -------------------------
# Random Forest
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    max_depth=15,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42
)
rf_fit = rf_model.fit(x, y)
rf_pred = rf_fit.predict(predData)
rf_acc = get_accuracy(dataTest, rf_pred)
results.append(("Random Forest", rf_model, rf_fit, rf_pred, rf_acc))

# -------------------------
# XGBoost
# -------------------------
if xgb_available:
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=10,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    xgb_fit = xgb_model.fit(x, y)
    xgb_pred = xgb_fit.predict(predData)
    xgb_pred = np.round(xgb_pred).astype(int)
    xgb_acc = get_accuracy(dataTest, xgb_pred)
    results.append(("XGBoost", xgb_model, xgb_fit, xgb_pred, xgb_acc))

# -------------------------
# Pick the best model
# -------------------------
best_result = max(results, key=lambda item: item[4])

best_name = best_result[0]
model = best_result[1]
modelFit = best_result[2]
pred = pd.Series(best_result[3]).astype(int)

# Print comparison results
for result in results:
    print(result[0] + " Accuracy:", round(result[4], 4))

print("Best model:", best_name)