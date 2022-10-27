"""
Script to collect data on the fly (for inference). Scrapes game lists from the
web and then uses NBA api to to collect data for the teams playing today.

@author: Riley Smith
Created: 10-31-2021
"""
from datetime import datetime

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

from sportsipy.nba.boxscore import Boxscore
import sportsipy.nba.schedule as schedule

TEAM_ABBREVIATIONS = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Lions': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

DROP_COLUMNS = ['location',
                 'losing_abbr',
                 'losing_name',
                 'winning_abbr',
                 'winning_name']

def list_available_games(game_date):
    """
    Find all the scheduled games for the given date

    Parameters
    ----------
    game_date : datetime object or str
        A datetime object for the date to get the schedule for or a string in
        the format '%Y%m%d'.
    """
    if isinstance(game_date, str):
        url = f'https://www.espn.com/nba/schedule/_/date/{game_date}'
    else:
        url = f'https://www.espn.com/nba/schedule/_/date/{game_date.strftime("%Y%m%d")}'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, features='lxml')
    # Get the first table (the one for the date of interest)
    table = soup.find_all('table', class_='schedule')[0]
    # Get the relevant rows
    rows = table.find('tbody').find_all('tr')
    # Populate list of available games
    available_games = []
    for row in rows:
        away_td, home_td = row.find_all('td')[:2]
        away = away_td.find('abbr')['title']
        home = home_td.find('abbr')['title']
        available_games.append({'home': TEAM_ABBREVIATIONS[home], 'away': TEAM_ABBREVIATIONS[away]})
    return available_games

def fetch_recent_data(team, game_date, n_values=[5, 10, 15]):
    """
    Get the recent game data for the given team as of the given game_date.

    Parameters
    ----------
    team : str
        The abbreviation for the team of interest.
    game_date : datetime object or str
        A datetime object for the date to get the schedule for or a string in
        the format '%m/%d/%Y'.
    """
    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date, '%m/%d/%Y')
    # Get the data for the given team in the given season
    year = game_date.year + 1 if game_date.month > 7 else game_date.year
    sched = schedule.Schedule(team, year=year).dataframe
    # Drop games which have no data yet
    sched = sched[~sched['losses'].isnull()]
    # Format date as datetime
    sched['datetime'] = pd.to_datetime(sched['datetime'], format='%Y-%m-%d')
    # Sort by date
    sched = sched.sort_values('date', ascending=False)
    # Grab only last n for maximum n
    sched = sched.head(max(n_values))
    # Get game data for each game
    game_data = []
    for idx, row in sched.iterrows():
        team_status = row['location']
        boxscore_id = row['boxscore_index']
        game_df = Boxscore(boxscore_id).dataframe.drop(DROP_COLUMNS, axis=1)
        game_df['team_status'] = team_status
        game_data.append(game_df)

        # game_df = game.boxscore.dataframe
        # if game_df is not None:
        #     game_data.append(game_df.drop(DROP_COLUMNS, axis=1))
        # else:
        #     break
    season_data = pd.concat(game_data, axis=0)
    season_data['date'] = pd.to_datetime(season_data['date'], format='%I:%M %p, %B %d, %Y')

    # return season_data
    # Get stats like last N
    relevant_data = {}
    for idx, row in season_data.iterrows():
        team_status = row['team_status']
        relevant_cols = [c for c in row.index if team_status.lower() in c]
        # Figure out winner of game
        won_game = 1 if row['winner'] == team_status else 0
        out_dict = {k[5:]: row[k] for k in relevant_cols}
        # out_dict = {k: row[k] for k in relevant_cols}
        out_dict['date'] = row['date']
        out_dict['win_pct'] = won_game
        relevant_data[idx] = out_dict

    # Make relevant data in DataFrame
    relevant_data = pd.DataFrame.from_dict(relevant_data, orient='index')
    relevant_data = relevant_data.sort_values('date', ascending=False)

    # return relevant_data

    last_n_dfs = []
    for n in n_values:
        last_n = relevant_data.head(n).drop(['date'], axis=1).mean(axis=0)
        last_n = last_n.rename(lambda x: x + f'_last_{n}')
        last_n_dfs.append(last_n)
    out_df = pd.concat(last_n_dfs)

    return out_df

def format_game_data(home_team, away_team, game_date, quiet=False):
    """
    Get the recent data for both home and away team and prepare it to go into
    the classifier.

    Parameters
    ----------
    home_team, away_team : str
        The abbreviations for home and away teams, respectively.
    game_date : datetime object or str
        A datetime object for the date to get the schedule for or a string in
        the format '%m/%d/%Y'.
    """
    print(f'{away_team} at {home_team}')
    if not quiet:
        print(f'\tGetting data for {home_team}')
    home_data = fetch_recent_data(home_team, game_date)
    if not quiet:
        print(f'\tGetting data for {away_team}')
    away_data = fetch_recent_data(away_team, game_date)
    # Rename to prepend column names with 'home_' and 'away_'
    home_data = home_data.rename(lambda x: 'home_' + x)
    away_data = away_data.rename(lambda x: 'away_' + x)
    # Concatenate into one array with home data coming first
    input_data = np.concatenate([home_data.to_numpy(), away_data.to_numpy()])
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

if __name__ == '__main__':
    ### METHOD FOR TESTING ###
    from datetime import timedelta
    gd = datetime.today() + timedelta(days=1)

    games = list_available_games(gd)
    print('games: ', games)
