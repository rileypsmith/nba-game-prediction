"""
New data pipeline serving data from a team focused perspective instead of a
player focused perspective. As a result, is much simpler and easier to read.

@author: Riley Smith
Created: 9-30-2021
"""
import pandas as pd

from team_based_game import TEAM_ABBREVIATIONS
TEAM_ABBREVIATIONS += ['BRK', 'CHO', 'NOP']

class NBADataPipeline():
    """
    A class for loading the data and then for performing various functions on
    it, like preprocessing/transforming it and running cross validation on a
    classifier.
    """
    def __init__(self, data_csv):
        self.data = self._load_data(data_csv)

    def _load_data(self, data_csv):
        """
        Load the data from a csv file. Just return the X and y, splitting into
        train and test data will be handled elsewhere.
        """
        # Load the data
        data = pd.read_csv(data_csv, index_col=0)
        # Turn truth (HOME/AWAY) into binary outcome variable
        data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)
        # Convert datetime string to datetime objects
        data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')
        return data.drop('winner', axis=1)

    def _preprocess_data(self):
        """Generate new features in the data that can be used for analysis."""
        raise NotImplementedError('Method not implemented.')

    def _scale(self):
        """Scale the data, in place"""
        raise NotImplementedError('Method not implemented.')
