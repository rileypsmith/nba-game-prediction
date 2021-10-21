"""
Simple functions to plot a roc curve.

@author: Riley Smith
Created: 10-16-2021
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_roc(fpr, tpr, thresholds, saveas=None):
    """
    Takes ndarrays for false positive rate and true positive rate, both ndarrays,
    as input and plots the corresponding ROC curve.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    # Plot line y=x
    x = np.linspace(0, 1, 2)
    ax.plot(x, x, color='dodgerblue', linestyle='--', label='Coin Flip')
    # Plot ROC points
    ax.plot(fpr, tpr, color='orange', label='ROC')
    ax.plot(fpr, thresholds, color='crimson', linestyle='dotted', label='Thresholds')
    # Make legend
    plt.legend()
    # Optionally save it
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()

def plot_kfolds_roc(fpr, tpr, saveas=None):
    """
    Take a list of false positive rates and true positive rates (one set for each
    of k folds from cross validation) and plot them all to see the variance
    in fitting to different folds.
    """
    fig, ax = plt.subplots(figsize=(8,8))

    # Plot line y=x
    x = np.linspace(0, 1, 2)
    ax.plot(x, x, color='dodgerblue', linestyle='--', label='Coin Flip')
    # Plot results for each fold
    for fpr_fold, tpr_fold in zip(fpr, tpr):
        ax.plot(fpr_fold, tpr_fold, color='navajowhite')
    # Make legend
    plt.legend()
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()
