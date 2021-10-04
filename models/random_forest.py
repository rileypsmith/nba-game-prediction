"""
A random forest classifier for predicting the outcome of NBA games.

@author: Riley Smith
Created: 8-18-2021
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.metrics import f1_score

import sys
sys.path.append('../data')
from data_pipeline import NBADataPipeline
import utils


if __name__ == '__main__':
    clf = Forest(n_estimators=200, min_samples_split=0.05, min_samples_leaf=0.01)

    data_pipeline = NBADataPipeline(test_pct=0, db_file='games.hdf5')
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = data_pipeline.get_data()

    refined_fields = [
        'PLUS_MINUS', 'EXP', 'FG_PCT', 'PTS', 'DREB', 'TOV', 'FG3_PCT', 'AST'
    ]

    train_data = utils.flatten(utils.get_fields(train_data, refined_fields))
    val_data = utils.flatten(utils.get_fields(val_data, refined_fields))

    print('Data stats:')
    print('\tTrain data shape: ', train_data.shape)
    print('\tTrain labels: ', train_labels.shape)
    print('\tVal data shape: ', val_data.shape)
    print('\tVal labels: ', val_labels.shape)
    print()

    print('Fitting classifier:')
    clf.fit(train_data, train_labels)

    preds = clf.predict(val_data)
    wrong = np.abs(preds - val_labels)
    wrong.sum() / wrong.size

    lost_predicted = np.where(preds == 0)
    wrong_lost_pred = wrong[lost_predicted]
    wrong_lost_pred.sum() / wrong_lost_pred.size

    val_labels.mean()

    f1_score(val_labels, np.ones_like(val_labels), pos_label=None, average='weighted')
    f1_score(val_labels, preds, pos_label=None, average='weighted')
