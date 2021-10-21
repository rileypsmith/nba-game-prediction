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

def build_dataset(year, output_csv='data/games.csv', resume=True):
    """
    Build a dataset for the given year using the sportsipy api. The dataset will
    consist of team, instead of player, statistics.

    Parameters
    ----------
    year : str
        The year to get data for. Should be the year that the season ended.
        For instance, '2021' would get data for the '2020-21' season.
    output_csv : str or os.pathlike
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
            output_csv = Path(output_csv)
            if not output_csv.exists():
                season_data.to_csv(str(output_csv), mode='w+', header=True)
            else:
                season_data.to_csv(str(output_csv), mode='a', header=False)
            # Reset game data for next team
            game_data = []
        # Save checkpoint by writing the done list to json
        with open('data/done_games.json', 'w+') as file:
            json.dump(done, file)

def fix_old_team_abbrs(data_csv):
    """Fix abbreviation for teams that changed their name"""
    # Load the data
    data = pd.read_csv(data_csv, index_col=0)
    data.replace('NOH', 'NOP', inplace=True)
    data.replace('CHA', 'CHO', inplace=True)
    data.replace('NJN', 'BRK', inplace=True)
    data.to_csv(data_csv)

def get_last_n_features(data_csv, n_values, output_csv=None, keep_original=False):
    """
    Get the average of features over the last 'n' games leading up to each game.

    Parameters
    ----------
    data_csv : str or os.pathlike
        The path to the .csv file containing all the necessary data.
    n_values : list or int
        If list, will separately compute the feature average over the last 'n'
        games for each value 'n' in the list. If an int, will compute it just
        for that value of 'n'.
    output_csv : str or os.pathlike
        The path to write the output data to. If None, uses the same as the
        input CSV.
    keep_original : bool
        If True, keep the data as it originally was and just concatenate the
        last 'n' data. If False, use only the last 'n' data and discard the
        original data.

    Returns
    -------
    None. Just writes the new output to csv.
    """
    # Make n a list if it comes in as an int
    if not isinstance(n_values, list):
        n_values = [n_values]
    # Read the data
    new_data = {}
    existing_data = pd.read_csv(data_csv, index_col=0)
    data = existing_data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data['outcome'] = data.apply(lambda x: 0 if x['winner'] == 'Away' else 1, axis=1)
    data = data.drop('winner', axis=1)

    for season in np.unique(data['season']):
        print('Working on', season)
        for team in tqdm(np.unique(data['home_team'])):
            # Get data for this team in the given season
            local_data = data[(data['season'] == season)
                                & ((data['home_team'] == team) | (data['away_team'] == team))]
            # Grab relevant data in each game for this team (home data if the
            # team is the home team, away data otherwise)
            relevant_data = {}
            for idx, row in local_data.iterrows():
                team_status = 'home' if row['home_team'] == team else 'away'
                relevant_cols = [c for c in row.index if team_status in c] + ['outcome']
                # Figure out winner of game
                won_game = row['outcome'] if team_status == 'home' else 1 - row['outcome']
                out_dict = {k.replace(team_status + '_', ''): row[k] for k in relevant_cols}
                out_dict['date'] = row['date']
                out_dict['win_pct'] = won_game
                relevant_data[idx] = out_dict
            # Make relevant data in DataFrame
            relevant_data = pd.DataFrame.from_dict(relevant_data, orient='index')
            # Sort it by date
            try:
                relevant_data = relevant_data.sort_values('date', ascending=True)
            except KeyError as ke:
                import pdb
                pdb.set_trace()
            # Apply a window to average
            last_n_dfs = []
            for n in n_values:
                last_n = relevant_data.drop(['date', 'outcome'], axis=1)\
                                      .rolling(n, min_periods=1).mean()
                last_n = last_n.rename(lambda x: x + f'_last_{n}', axis=1)
                last_n_dfs.append(last_n)
            # Concatenate them or just save it if there is only 1
            if len(last_n_dfs) == 1:
                out_df = last_n_dfs[0]
            else:
                out_df = pd.concat(last_n_dfs, axis=1)
            # Store it
            for idx, row in local_data.iterrows():
                team_status = 'home' if row['home_team'] == team else 'away'
                if idx in new_data:
                    new_data[idx][team_status] = out_df.loc[idx]
                else:
                    new_data[idx] = {team_status: out_df.loc[idx]}

    # Now go through dictionary and concatenate home and away data into single df
    out_dict = {}
    for game_idx, last_n_dict in new_data.items():
        away_data = last_n_dict['away'].rename(lambda x: 'away_' + x)
        home_data = last_n_dict['home'].rename(lambda x: 'home_' + x)
        full_last_n = pd.concat([away_data, home_data], axis=0)
        out_dict[game_idx] = full_last_n.to_dict()

    # Finally turn them all into a dataframe and sort by index
    final_out_df = pd.DataFrame.from_dict(out_dict, orient='index')
    # final_out_df = pd.concat([out_dict[k] for k in out_dict], axis=0)
    final_out_df = final_out_df.sort_index(axis=0)

    if keep_original:
        # Sort original data by index and concatenate columns
        existing_data = existing_data.sort_index(axis=0)
        final_out_df = pd.concat([existing_data, final_out_df], axis=1)
    else:
        metadata = existing_data[['home_team', 'away_team', 'date', 'winner']]
        metadata = metadata.sort_index(axis=0)
        final_out_df = pd.concat([final_out_df, metadata], axis=1)


    if output_csv is not None:
        final_out_df.to_csv(output_csv)
    else:
        final_out_df.to_csv(data_csv)

def label_home_team(data_csv, output_csv=None):
    """
    Loop through the data and add home team and away team columns retrospectively.
    """
    if output_csv = None:
        output_csv = data_csv
    existing_data = pd.read_csv(data_csv, index_col=0)
    # Set new columns
    existing_data['home_team'] = ['none']*existing_data.index.size
    existing_data['away_team'] = ['none']*existing_data.index.size
    existing_data['season'] = np.zeros((existing_data.index.size,)) * np.nan
    existing_data['date'] = ['none']*existing_data.index.size

    for season in range(2010, 2022, 1):
        done = []
        print('Working on', season)
        nets_abbr = 'NJN' if season < 2013 else 'BRK'
        cha_abbr = 'CHA' if season < 2015 else 'CHO'
        pels_abbr = 'NOH' if season < 2014 else 'NOP'
        for team in tqdm(TEAM_ABBREVIATIONS + [nets_abbr, cha_abbr, pels_abbr]):
            sched = schedule.Schedule(team, year=season).dataframe
            if team == 'NJN':
                team = 'BRK'
            elif team == 'CHA':
                team = 'CHO'
            elif team == 'NOH':
                team = 'NOP'
            for boxscore_index, row in sched.iterrows():
                if boxscore_index in done:
                    continue
                home_team = team if row['location'] == 'Home' else row['opponent_abbr']
                away_team = team if row['location'] == 'Away' else row['opponent_abbr']
                game_date = row['datetime'].strftime('%Y-%m-%d')
                try:
                    existing_data.at[boxscore_index, 'home_team'] = home_team
                    existing_data.at[boxscore_index, 'away_team'] = away_team
                    existing_data.at[boxscore_index, 'season'] = season
                    existing_data.at[boxscore_index, 'date'] = game_date
                except Exception as e:
                    print(repr(e))
                    pass
                done.append(boxscore_index)
    existing_data.to_csv(output_csv)

if __name__ == '__main__':
    seasons = list(range(2010, 2022, 1))
    output_csv = 'data/games.csv'
    for season in seasons:
        build_dataset(season, output_csv=output_csv)
    label_home_team(output_csv)
    fix_old_team_abbrs(output_csv)
    get_last_n_features(output_csv, [5, 10, 15], output_csv='data/data.csv')
