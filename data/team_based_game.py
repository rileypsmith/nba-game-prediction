"""
A script for a team focused (instead of player focused) approach to collecting
data (which seeks to collect features about the team performance instead of
the performance of individual players).

@author: Riley Smith
Created: 9-6-2021
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import sportsipy.nba.schedule as schedule

DROP_COLUMNS = ['date',
                 'location',
                 'losing_abbr',
                 'losing_name',
                 'winning_abbr',
                 'winning_name']

# TODO: Come up with a better way to store this. Not very clean to just have it
# in the .py file like this. Maybe JSON??
TEAM_ABBREVIATIONS = ['ATL',
                     'BOS',
                     'CHI',
                     'CLE',
                     'DAL',
                     'DEN',
                     'DET',
                     'GSW',
                     'HOU',
                     'IND',
                     'LAC',
                     'LAL',
                     'MEM',
                     'MIA',
                     'MIL',
                     'MIN',
                     'NYK',
                     'OKC',
                     'ORL',
                     'PHI',
                     'PHO',
                     'POR',
                     'SAC',
                     'SAS',
                     'TOR',
                     'UTA',
                     'WAS']

def build_dataset(year, output_file='data/games.csv', resume=True):
    """
    Build a dataset for the given year using the sportsipy api. The dataset will
    consist of team, instead of player, statistics.

    Parameters
    ----------
    year : str
        The year to get data for. Should be the year that the season ended.
        For instance, '2021' would get data for the '2020-21' season.
    output_file : str or os.pathlike
        The path to where output should be stored (should be a .csv file).
    resume : bool
        If True, will load the list of games already processed and start from
        there.
    """
    # Keep track of which games you have already gotten data for
    if resume:
        with open('data/done_games.json', 'r') as file:
            done = json.load(file)
    else:
        done = []
    # Loop over each team, grabbing data for each game
    game_data = []
    # Handle Brooklyn/New Jersey Nets and Charlotte Hornets/Bobcats
    nets_abbr = 'NJN' if year < 2013 else 'BRK'
    cha_abbr = 'CHA' if year < 2015 else 'CHO'
    pels_abbr = 'NOH' if year < 2014 else 'NOP'
    for team in TEAM_ABBREVIATIONS + [nets_abbr, cha_abbr, pels_abbr]:
        sched = schedule.Schedule(team, year=year)
        # Loop through the games and get all data
        for game in tqdm(sched):
            # If you have already seen this game, continue
            box_id = game.boxscore_index
            if box_id in done:
                continue
            # If not store it so you don't count it twice
            done.append(box_id)
            # Fetch all the data
            try:
                box_df = game.boxscore.dataframe.drop(DROP_COLUMNS, axis=1)
            except ZeroDivisionError as zde:
                time.sleep(2)
                continue
            time.sleep(2)
            # Store it
            game_data.append(box_df)
        # Stack all the data into a Pandas DataFrame
        if game_data:
            season_data = pd.concat(game_data, axis=0)
            # Append it to the csv file
            output_file = Path(output_file)
            if not output_file.exists():
                season_data.to_csv(str(output_file), mode='w+', header=True)
            else:
                season_data.to_csv(str(output_file), mode='a', header=False)
            # Reset game data for next team
            game_data = []
        # Save checkpoint by writing the done list to json
        with open('data/done_games.json', 'w+') as file:
            json.dump(done, file)

if __name__ == '__main__':
    # for season in range(2017, 2022, 1):
    #     build_dataset(season, resume=True)
    seasons = list(range(2010, 2022, 1))
    for season in seasons:
        build_dataset(season)
