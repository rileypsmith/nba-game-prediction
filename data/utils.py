"""
Utility functions for formatting NBA data.

@author: Riley Smith
Created: 5-28-2021
"""
from datetime import datetime, timedelta

def format_season(season_date, return_year=False):
    """
    Date is string of form "MM-DD-YYYY". Extracts season from this string.

    For example, "05-28-21" returns the season "2020-21".
    """
    if not isinstance(season_date, datetime):
        season_date = datetime.strptime(season_date, "%m-%d-%Y")
    starting_year = season_date.year if season_date.month > 9 else season_date.year - 1
    out_date_str = f'{starting_year}-{str(starting_year + 1)[2:]}'
    if return_year:
        return out_date_str, starting_year
    return out_state_str

def nba_dict(data):
    """
    Takes data dictionary from many of the NBA API endpoints and reformats it
    in a more usable way.
    """
    headers = data['headers']
    content = data['data'][0]
    return {k: v for k, v in zip(headers, content)}

def flatten(game_data):
    """Flatten the game data into a vector.

    Parameters
    ----------
    game_data : ndarray
        An ndarray of shape (# games, 2, 11, 52). 2 teams, 10 players + 1 team aggregate
        statistic, and 52 features.

    Returns
    -------
    flattened_data : ndarray
        An ndarray of shape (1144,). The flattened game data.
    """
    return game_data.reshape(game_data.shape[0], -1)

def data_labels():
    """
    A function to return a dictionary mapping each data index to its actual
    label (for interpreting which features are important).
    """
    data_fields = [
        'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
        'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS'
    ]
    tmp = ['USAGE']
    for item in data_fields:
        tmp += [item]*3
    tmp += ['HEIGHT', 'WEIGHT', 'EXP', 'UNDRAFTED']
    out = {}
    for i, item in enumerate(tmp):
        out[i] = item
    return out

def get_fields(data, fields):
    """
    Extract from the data only the named fields provided.

    Parameters
    ----------
    data : ndarray
        The data of shape (# games, 2, 11, 52) to filter
    fields : list
        A list of field names to keep.

    Returns
    -------
    filtered_data : ndarray
        The data only in the specified fields.
    """
    _data_labels = data_labels()
    # Figure out which indices go to which data fields
    field_indices = {}
    for idx, field in _data_labels.items():
        try:
            field_indices[field].append(idx)
        except KeyError as ke:
            field_indices[field] = [idx]
    # Get the indices for the fields of interest
    interest_indices = []
    for field in fields:
        interest_indices += field_indices[field]
    return data[:,:,:,interest_indices]

def get_outcome(game_info):
    """
    Get the score of the game based on one team's score and the plus minus.

    Parameters
    ----------
    game_info : list
        A list of the info for this game, coming form LeagueGameLog. From the
        perspective of the home team.

    Returns
    -------
    game_outcome : tuple
        The score for home (first entry in tuple) and away (second entry in tuple)
        and a binary indicator for whether or not the home team won (3rd entry).
    """
    home_pts = game_info[-3]
    plus_minus = game_info[-2]
    away_pts = home_pts - plus_minus
    home_win = 1 if home_pts > away_pts else 0
    return (home_pts, away_pts, home_win)
