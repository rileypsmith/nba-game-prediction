"""
New data pipeline serving data from a team focused perspective instead of a
player focused perspective. As a result, is much simpler and easier to read.

@author: Riley Smith
Created: 9-30-2021
"""
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from build_dataset import TEAM_ABBREVIATIONS
TEAM_ABBREVIATIONS += ['BRK', 'CHO', 'NOP']

class NBADataPipeline():
    """
    A class for loading the data and then for performing various functions on
    it, like preprocessing/transforming it and running cross validation on a
    classifier.
    """
    def __init__(self, data_csv, use_pca=True, pca_components=10, scale_data=True,
                    delete_first_ten=False, holdout_year=None):
        """
        Parameters
        ----------
        data_csv : str or os.pathlike
            The path to the CSV file containing data.
        use_pca : bool
            Whether or not to use PCA in order to reduce the dimensionality of
            the data.
        pca_components : int
            The number of principal components to retain in PCA.
        scale_data : bool
            Whether or not to apply standard scaling to data input (recommended
            to be True).
        delete_first_ten : bool
            If True, ignore first 10 games of each season (as a data warmup).
        holdout_year : int
            Optional year to withold as test data. If given, all games in that
            season become test data and the train data does not include any of
            those games. Year provided is when season started (i.e. year=2020
            would be the 2020-21 season).
        """
        self._X, self._y, self._X_test, self._y_test, self._meta = self._load_data(data_csv, delete_first_ten, holdout_year)

        self._pipeline = self._gen_pipeline(use_pca=use_pca,
                                            pca_components=pca_components,
                                            scale_data=scale_data)

    def _load_data(self, data_csv, delete_first_ten, holdout_year):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        # Load the data
        data = pd.read_csv(data_csv, index_col=0)
        # If deleting first ten games, get rid of all games where last_10 is
        # equal to last_15
        if delete_first_ten:
            data = data[data['home_points_last_15'] != data['home_points_last_10']]
        # Turn truth (HOME/AWAY) into binary outcome variable
        data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)
        # Convert datetime string to datetime objects
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        # If a holdout year is given, split that data off
        if holdout_year is not None:
            season_start = datetime(holdout_year, 10, 1)
            season_end = datetime(holdout_year + 1, 7, 1)
            holdout = data[(data['date'] < season_end)
                            & (data['date'] > season_start)]
            data = data[(data['date'] > season_end)
                            | (data['date'] < season_start)]
            y_test = holdout['outcome']
            X_test = holdout.drop(['winner', 'outcome', 'home_team', 'away_team', 'date'], axis=1)
            X_test = X_test.fillna(0)
            # Make them ndarrays
            y_test = y_test.to_numpy()
            X_test = X_test.to_numpy()
        else:
            X_test = None
            y_test = None
        # Split into X, y, and meta
        y = data['outcome']
        meta = data[['home_team', 'away_team', 'date']]
        X = data.drop(['winner', 'outcome', 'home_team', 'away_team', 'date'], axis=1)
        # Take care of any NaN values
        X = X.fillna(0)
        return X.to_numpy(), y.to_numpy(), X_test, y_test, meta

    def _gen_pipeline(self, use_pca=True, pca_components=10, scale_data=True):
        """
        Using the options provided, produce an Sklearn data pipeline object
        that can be fit to the data and later used to transform data at test
        time.

        Parameters
        ----------
        use_pca : bool
            Whether or not to run PCA on the data.
        pca_components : int
            How many components to use for PCA (irrelevant if use_pca is False).
        scale_data : bool
            Whether or not to do mean/standard deviation scaling of each feature.

        Returns
        -------
        Sklearn pipeline instance.
        """
        # Make list of preprocessing elements in pipeline
        pipeline_components = []
        if scale_data:
            scaler = StandardScaler()
            pipeline_components.append(('Standard Scaler', scaler))
        if use_pca:
            pca = PCA(pca_components)
            pipeline_components.append(('PCA', pca))
        return Pipeline(pipeline_components)

    def fit_pipeline(self):
        """Fit the pipeline to all the training data and pre-process test data"""
        self._X = self._pipeline.fit_transform(self._X)
        if self._X_test is not None:
            self._X_test = self._pipeline.transform(self._X_test)

    def preprocess(self, X):
        """Preprocess the incoming data for inference"""
        return self._pipeline.transform(X)

    def yield_kfolds(self, k):
        """Yield k folds for cross validation"""
        # Build KFold sklearn object
        kf = KFold(n_splits=k)
        # For each fold, preprocess data and then yield it
        for train_indices, test_indices in kf.split(self._X):
            # Grab train and test data and labels
            X_train, X_test = self._X[train_indices], self._X[test_indices]
            y_train, y_test = self._y[train_indices], self._y[test_indices]
            # Fit preprocessor to training data
            X_train = self._pipeline.fit_transform(X_train)
            # Transform test data
            X_test = self._pipeline.transform(X_test)
            # Yield it
            yield (X_train, X_test, y_train, y_test)

    @property
    def train_data(self):
        """
        Getter for returning the training data (X and y, does the same thing as
        the getter for just data).
        """
        return self._X, self._y

    @property
    def test_data(self):
        """
        Return the data for the holdout season if there is any, otherwise return
        None.
        """
        if self._X_test is not None:
            return self._X_test, self._y_test
        return None

    @property
    def pipeline(self):
        """Return the actual Sklearn pipeline instance"""
        return self._pipeline
