"""
Simple function to plot a roc curve.

@author: Riley Smith
Created: 10-16-2021
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_roc(fpr, tpr, thresholds):
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
    # ax.plot(baseline_fpr, baseline_tpr, color='mediumseagreen', label='Baseline')
    ax.plot(fpr, thresholds, color='crimson', linestyle='dotted', label='Thresholds')

    plt.legend()
    plt.show()

def plot_kfolds_roc(fpr, tpr):
    fig, ax = plt.subplots(figsize=(8,8))

    # Plot line y=x
    x = np.linspace(0, 1, 2)
    ax.plot(x, x, color='dodgerblue', linestyle='--', label='Coin Flip')

    for fpr_fold, tpr_fold in zip(fpr, tpr):
        ax.plot(fpr_fold, tpr_fold, color='navajowhite')
    # ax.plot(fpr.mean(axis=0), tpr.mean(axis=0), color='orange', label='Average ROC')

    plt.legend()
    plt.show()
