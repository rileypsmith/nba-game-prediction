"""
Script to build out a database of game data (for faster loading during training).

@author: Riley Smith
Created: 7-8-2021
"""
import h5py
import json
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from nba_api.stats.endpoints import LeagueGameLog

from utils import format_season, nba_dict, get_outcome
import player_info_db

class GameDB():
    """Database class for game-by-game data"""
    def __init__(self, db_name='data/player_db.hdf5'):
        self.db_name = db_name

    def build(self, date_range=(1980, 2020), reset=True):
        """Build the dataset for the given date range"""
        # Open HDF5 file
        db_file = Path(self.db_name)
        if not db_file.exists():
            # Prompt user if they would like to create the database
            make_dataset = input(f'File {self.db_name} does not exist. Would you like to create it? [Y/N]')
            if make_dataset.upper() != 'Y':
                print('Exiting')
                sys.exit(0)
            else:
                # Go ahead and make the hdf5 file
                db_file.touch()

        # Build for all seasons in the date range
        for season in range(date_range[0], date_range[1], 1):
            print(f'Getting game data for {season}')

            with h5py.File(self.db_name, 'a') as db:
                # Create a group for this season in the hdf5 file if it does not
                # already exist
                if not str(season) in db.keys():
                    db.create_group(str(season))
                    finished = np.array([0])
                    db.create_dataset(f'{season}/finished', data=finished)
                # Goup does exist, but reset the finished counter
                if reset:
                    finished = np.array([0])
                    if 'finished' not in db[str(season)].keys():
                        db.create_dataset(f'{season}/finished', data=finished)
                    else:
                        db_finished = db[f'{season}/finished']
                        db_finished[...] = finished
                else:
                    finished = db[f'{season}/finished']
                    finished = finished[0]

            # Ping API for all games that year
            games = LeagueGameLog(season=season).league_game_log.get_dict()['data']
            games = [g[4] for g in games]
            time.sleep(1.8)

            # Setup progress bar manually
            pbar = tqdm(total=len(games)//2, initial=int(finished))
            pbar.set_description(f'Season - {season}')

            for game_id in games[::2][int(finished):]:
                # Create a game object for this game, which will grab the needed data
                try:
                    game_obj = Game(game_id)
                except json.decoder.JSONDecodeError as e:
                    time.sleep(1.8)
                    continue
                time.sleep(1.8)

                # Store data in the database
                with h5py.File(self.db_name, 'a') as file:
                    # Get home and away data
                    home_data = [[p[0]] + p[1] for p in game_obj.player_stats['home']]
                    away_data = [[p[0]] + p[1] for p in game_obj.player_stats['away']]
                    # Create datasets for them in hdf5 file
                    if len(home_data) > 0:
                        try:
                            home_dset = file.create_dataset(f'{season}/{game_id}/home', data=home_data)
                            away_dset = file.create_dataset(f'{season}/{game_id}/away', data=away_data)
                        except:
                            if reset:
                                raise ValueError(f'Datasets already exist for {season}/{game_id}.')

                    # Update the number of games finished
                    finished += 1
                    db_finished = file[f'{season}/finished']
                    db_finished[...] = finished

                # Update progress bar
                pbar.update(1)
                pbar.set_description(f'Season - {season}')

    def get_labels(date_range=(2011,2020)):
        """
        Label each game according to who won and the score.

        Parameters
        ----------
        date_range : tuple
            A tuple of integers specifying the date range (seasons) for which to
            grab game outcome information.

        Returns
        -------
        None. Just labels each game in the given db file.
        """
        for season in tqdm(range(date_range[0], date_range[1], 1)):
            # Get games from LeagueGameLog
            games = LeagueGameLog(season=season).league_game_log.get_dict()['data']
            time.sleep(1.8)
            # Get only the games from the home perspective
            home_perspective = [g for g in games if 'vs.' in g[6]]
            # Make empty game outcomes dict
            game_outcomes = {}
            # For each one, get the game outcome
            print('Formatting game outcomes')
            for row in tqdm(home_perspective):
                outcome = get_outcome(row)
                game_outcomes[row[4]] = outcome
            # Store each outcome in hdf5 file
            with h5py.File(self.db_name, 'a') as file:
                for key in game_outcomes:
                    if not key in file[str(season)].keys():
                        continue
                    file.create_dataset(f'{season}/{key}/outcome', data=np.array(game_outcomes[key]))
        return


if __name__ == '__main__':
    db = GameDB()
    db.build(date_range=(2012, 2020), reset=True)
    db.get_labels(date_range=(2012, 2020))
