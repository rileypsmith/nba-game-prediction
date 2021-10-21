"""
An all-encompassing NBA game data pipeline for serving up data from the hdf5
file.

NOTE (10-21-21): This was originally written intending to use player-level data
as the focus. It has now been replaced by one focusing on team level data (see
data_pepline.py).

@author: Riley Smith
Created: 8-14-2021
"""
import h5py

import numpy as np
from tqdm import tqdm

class NBADataPipeline():
    def __init__(self, val_pct=0.1, test_pct=0.1, seasons='all', db_file='data/games.hdf5'):
        """
        Parameters
        ----------
        val_pct, test_pct : float
            In range [0,1]. The percentage of data to use for validation and test
            datasets, respectively.
        seasons : list or str
            If str, must be 'all', meaning use all available seasons. Otherwise
            a list of season integers to use.
        """
        # Set data constants
        self.db_file = db_file
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.seasons = seasons

    def _pad_dataset(self, dataset, game_id, season, key):
        """Take the dataset form the hdf5 file and pad it to have 10 rows"""
        pad_dims = ((0, 10 - dataset.shape[0]), (0, 0))
        return np.pad(dataset, pad_dims)

    def team_aggregate_statistic(self, team_data):
        """
        Generate feature by summing up the stats for a team, weighted by
        minutes played in the game.

        Parameters:
        -----------
        team_data : ndarray
            An ndarray of shape (10, 52). The data for all 10 players on the team
            that were relevant in this game.
        """
        # Split off overall usage stat and basic player info
        trimmed_data = team_data[:,1:-3]
        # Split off minutes
        minutes = trimmed_data[:,:3]
        # Tile minutes so it can easily be multiplied by the rest of the data
        minutes = np.tile(minutes, (1, 16))
        # Weight data by minutes, divided by 240 (total number of minutes played in an NBA game)
        weighted_data = (trimmed_data * minutes) / 240
        # Sum over that data
        weighted_sum = weighted_data.sum(axis=0)
        # Pad with zeros for stats that it does not make sense to sum over
        weighted_sum = np.pad(weighted_sum, (1, 3))
        # Add it onto data
        output = np.concatenate([team_data, np.expand_dims(weighted_sum, axis=0)], axis=0)
        return output

    def get_data(self, random_state=1234):
        """Return the training, validation and test datasets"""
        # Make empty container lists for all data
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        test_data = []
        test_labels = []

        # Open hdf5 file
        with h5py.File(self.db_file, 'r') as db:
            if self.seasons == 'all':
                seasons = db.keys()
            else:
                seasons = self.seasons

            # For each season, load data for each game
            for season in tqdm(seasons):
                # Make hdf5 group
                season_group = db[season]

                # Make blank lists for all data this season
                season_data = []
                season_labels = []

                # Grab data for each game
                for game in season_group.keys():
                    if game == 'finished':
                        continue
                    # Pad home and away data
                    home_data = self._pad_dataset(season_group[game]['home'][...], game, season, 'home')
                    away_data = self._pad_dataset(season_group[game]['away'][...], game, season, 'away')
                    # Get team aggregate stat for each team
                    home_data = self.team_aggregate_statistic(home_data)
                    away_data = self.team_aggregate_statistic(away_data)
                    # Stack into full game data and get outcome
                    game_data = np.stack([home_data, away_data], axis=0)
                    outcome = season_group[game]['outcome'][...][-1]
                    # Store data
                    season_data.append(game_data)
                    season_labels.append(outcome)
                # Stack all the games' data into one season dataset
                season_data = np.stack(season_data, axis=0)
                season_labels = np.array(season_labels)
                # Separate out the validation and test data randomly
                rng = np.random.default_rng(random_state)
                indices = np.arange(season_labels.size)
                np.random.shuffle(indices)
                # Get valid data
                num_valid = round(self.val_pct * indices.size)
                local_val_data = season_data[:num_valid]
                local_val_labels = season_labels[:num_valid]
                val_data.append(local_val_data)
                val_labels.append(local_val_labels)
                used = num_valid
                # Get test data, if test data is called for
                if self.test_pct is not None and self.test_pct > 0:
                    num_test = round(self.test_pct * indices.size)
                    local_test_data = season_data[used:used + num_test]
                    local_test_labels = season_labels[used:used + num_test]
                    test_data.append(local_test_data)
                    test_labels.append(local_test_labels)
                    used += num_test
                # Get training data
                local_train_data = season_data[used:]
                local_train_labels = season_labels[used:]
                train_data.append(local_train_data)
                train_labels.append(local_train_labels)

        # Concatenate all the data into large ndarrays and return them
        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        val_data = np.concatenate(val_data, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        if test_data:
            test_data = np.concatenate(test_data, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

if __name__ == '__main__':
    data_pipeline = NBADataPipeline(test_pct=0)
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = data_pipeline.get_data()
