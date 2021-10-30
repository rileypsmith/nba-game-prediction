"""
A logistic regression classifier for predicting the outcome of NBA games.

@author: Riley Smith
Created: 10-21-2021
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tqdm import tqdm

import evaluation as eval

import sys
sys.path.append('../data')
sys.path.append('../visualization')
from data_pipeline import NBADataPipeline
from roc_curve import plot_roc, plot_kfolds_roc

# Build basic logistic regression classifier
clf = LogisticRegression()

# Get data
data_csv = r"C:\Users\thehu\OneDrive\Documents\Code_Projects\nba-game-simulator\data\data.csv"
data_pipeline = NBADataPipeline(data_csv, delete_first_ten=True)

# Try fitting it and examining results
running_fpr = []
running_tpr = []
for X_train, X_test, y_train, y_test in tqdm(data_pipeline.yield_kfolds(5), total=5):
    # Fit classifier
    clf.fit(X_train, y_train)
    # Get predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    # Get ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob[:,-1])
    # Store results
    running_fpr.append(fpr)
    running_tpr.append(tpr)

plot_kfolds_roc(running_fpr, running_tpr)

############################################################
#  Now that we have basic results, let's do a grid search  #
############################################################

# Build custom model evaluation based on false positive rate
fpr_eval = eval.FixedFPREvaluation(0.15)

# Make dictionary to store output results
grid_search_results = {}

# Set grid search parameters
params = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.5, 1., 2., 4.],
    'solver': ['saga'],
    'max_iter': [100, 200],
}

# Try different values for PCA
for k in range(10, 110, 10):
    print('\n\nWorking on', k, 'components...\n')
    # Get data for this particular PCA projection
    data_pipeline = NBADataPipeline(data_csv, pca_components=k, delete_first_ten=True)
    X, y = data_pipeline.train_data
    # For each PCA value, do a grid search to find best Random Forest parameters
    reg = LogisticRegression()
    clf = GridSearchCV(reg, params, scoring=fpr_eval, verbose=1)
    clf.fit(X, y)
    # Store results
    local_results = {
        'estimator': clf.best_estimator_,
        'score': clf.best_score_,
        'params': clf.best_params_,
        'time': clf.refit_time_
    }
    grid_search_results[k] = local_results

# Store it to JSON so you can recover the best results later
to_json = {}
for k in grid_search_results:
    d = grid_search_results[k]
    to_json[k] = {
        'params': d['params'],
        'score': d['score']
    }
with open('logistic_regression_best_results.json', 'w+') as file:
    json.dump(to_json, file)

############################################################
#            And now we can evaluate our model             #
############################################################

# Now let's get the best model and see how it performs on an entire hold out year
sorted_results = sorted(grid_search_results.items(), key=lambda x: x[1]['score'], reverse=True)
best_k = sorted_results[0][0]
best_params = sorted_results[0][1]['params']

# Build data pipeline
data_pipeline = NBADataPipeline(data_csv, pca_components=best_k, delete_first_ten=True,
                                holdout_year=2019)
data_pipeline.fit_pipeline()

# Built classifier with best parameters
clf = LogisticRegression(**best_params)

# Fit it to the data
X_train, y_train = data_pipeline.train_data
clf.fit(X_train, y_train)

# Get the test data
X_test, y_test = data_pipeline.test_data

# Now let's see how we did. First plot a roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:,-1])
thresholds = np.clip(thresholds, 0, 1)
plot_roc(fpr, tpr, thresholds)

# Now make an accuracy plot
accuracies = []
for threshold in np.linspace(0, 1, 100):
    accuracies.append(eval.ThresholdedAccuracy(threshold)(clf, X_test, y_test))
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel('Prediction probability threshold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Logistic Regression Best Accuracy')
ax.plot(np.linspace(0, 1, 100), np.array(accuracies) * 100)
plt.show()
