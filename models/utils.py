"""
Utils for building models.

@author: Riley Smith
Created: 10-17-2021
"""
import numpy as np
from sklearn import metrics

class FixedFPREvaluation():
    def __init__(self, fpr):
        """
        Given a set false positive rate (fpr), evaluate the model based on how
        many true positives it gets while staying below that false positive
        rate.
        """
        self._fpr = fpr

    def __call__(self, estimator, X, y):
        """
        Assumes you are dealing with binary classification.

        Parameters
        ----------
        estimator : Sklearn estimator
            The estimator to evaluate.
        X : ndarray
            Validation data of shape (n_samples, n_features).
        y : ndarray
            Ground truth for X. Has shape (n_samples,)
        """
        # Get the probabilities from the estimator
        prob = estimator.predict_proba(X)
        # Get ROC curve for predicting home team first
        home_fpr, home_tpr, _ = metrics.roc_curve(y, prob[:,-1])
        # Find where fpr is below threshold and get max tpr there
        max_home_tpr = home_tpr[np.where(home_fpr < self._fpr)].max()
        # Weight by number of games in which home team won
        max_home_tpr *= y.mean()
        # Now do the same for away
        away_fpr, away_tpr, _ = metrics.roc_curve(1 - y, prob[:,0])
        max_away_tpr = away_tpr[np.where(away_fpr < self._fpr)].max()
        max_away_tpr *= (1 - y.mean())

        return max_home_tpr + max_away_tpr

class ThresholdedAccuracy():
    """
    Evaluate a classifier based on its accuracy above a certain threshold.
    """
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, estimator, X, y):
        """
        Assumes you are dealing with binary classification.

        Parameters
        ----------
        estimator : Sklearn estimator
            The estimator to evaluate.
        X : ndarray
            Validation data of shape (n_samples, n_features).
        y : ndarray
            Ground truth for X. Has shape (n_samples,)
        """
        # Get predictions from estimator
        preds = estimator.predict(X)
        # Get the probabilities from the estimator
        prob = estimator.predict_proba(X)
        # Grab only those that are above the threshold
        home_refined_pred = preds[np.where(prob[:,-1] > self._threshold)]
        home_refined_true = y[np.where(prob[:,-1] > self._threshold)]
        # Compute accuracy and weight by number of surviving games
        home_acc = metrics.accuracy_score(home_refined_true, home_refined_pred)
        home_acc *= home_refined_pred.size
        # Do the same for away
        away_refined_pred = preds[np.where(prob[:,0] > self._threshold)]
        away_refined_true = y[np.where(prob[:,0] > self._threshold)]
        away_acc = metrics.accuracy_score(1 - away_refined_true, away_refined_pred)
        away_acc *= away_refined_pred.size
        # Return the weighted average
        return (home_acc + away_acc) / (home_refined_pred.size + away_refined_pred.size)
