"""
Visualize the correlation between different featuers to get some idea of which
features may be redundant.

@author: Riley Smith
Created: 10-3-2021
"""
import numpy as np

import matplotlib.pyplot as plt

def show_correlations(data, saveas=None):
    """
    Plot the correlation matrix between features.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame where each row is a different game and each column is a
        feature.
    """
    # Drop column for winner
    working_data = data.drop('winner', axis=1).fillna(0)
    # Compute correlatiosn
    corr = np.corrcoef(working_data.to_numpy(), rowvar=False)
    # Plot it
    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(corr, cmap='plasma')
    # Set ticks
    ax.set_xticks(np.arange(working_data.columns.size))
    ax.set_yticks(np.arange(working_data.columns.size))
    ax.set_xticklabels(working_data.columns, fontsize=8, rotation=90)
    ax.set_yticklabels(working_data.columns, fontsize=8)
    plt.colorbar(im, ax=ax)
    # Optionally save it
    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('data/games.csv', index_col=0)
    show_correlations(data)
