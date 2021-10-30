"""
Visualization for the portion of explained variance for different numbers of
principal components retained.

@author: Riley Smith
Created: 10-23-2021
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import sys
sys.path.append('../data')
from data_pipeline import NBADataPipeline

def plot_explained_variance(data_pipeline, saveas=None):
    """
    Run PCA on the training data in the pipeline and plot explained variance
    as a function of retained principal components.
    """
    # Get data
    X, _ = data_pipeline.train_data
    # Fit it to a PCA with all components
    pca = PCA(X.shape[1])
    pca.fit(X)
    # Plot explained var
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    plot_x = np.arange(X.shape[1])
    plot_y = pca.explained_variance_ratio_
    ax[0].plot(plot_x, plot_y, color='orange', label='Explained Variance')
    ax[1].plot(plot_x, np.cumsum(plot_y), color='dodgerblue', label='Cumulative Explained Variance')
    ax[0].set_title('Explained variance for individual\nprincipal components')
    ax[1].set_title('Cumulative explained variance')

    fig.text(0.5, 0.04, 'Number of principal components', ha='center')
    ax[0].set_ylabel('Explained variance')
    # plt.legend()
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()
