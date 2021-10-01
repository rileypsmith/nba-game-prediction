"""
A script to setup a local SQLite db for storing basic player information (for
quicker lookup, so you don't have to use the NBA API each time).

@author: Riley Smith
Created: 6-17-2021
"""
__author__ = 'Riley Smith'

import sqlite3 as sqlite
from time import sleep
import warnings

from tqdm import tqdm

from nba_api.stats.endpoints import CommonPlayerInfo
from nba_api.stats.endpoints import CommonAllPlayers

from utils import nba_dict

def read(player_id, db_name='data/player_info.db'):
    """Read the player info for the given player id"""
    # Connect to db
    conn = sqlite.connect(db_name)
    cursor = conn.cursor()
    # Retrieve rows
    sql_query = f"""SELECT * FROM player_info WHERE player_id = ?"""
    rowset = cursor.execute(sql_query, (player_id,))

    output = [list(row) for row in rowset]

    conn.close()

    return output[0]

class PlayerInfoDB():
    def __init__(self, db_name='data/player_info.db'):
        self.db_name = db_name

    def _individual_player_info(self, player_id):
        """Gets height, weight, experience just for the given player"""
        # Retrieve player data from NBA API
        player_info = CommonPlayerInfo(player_id)
        player_info = nba_dict(player_info.common_player_info.get_dict())

        # Extract and format height
        if player_info['HEIGHT'] == '':
            warnings.warn(f'Missing height for player {player_info["FIRST_NAME"]} {player_info["LAST_NAME"]}. Setting to 0 instead.')
            height = 0.
        else:
            ft, inches = player_info['HEIGHT'].split('-')
            height = float(ft) + (float(inches) / 12)

        # Format weight
        if player_info['WEIGHT'] == '':
            warnings.warn(f'Missing weight for player {player_info["FIRST_NAME"]} {player_info["LAST_NAME"]}. Setting to 0 instead.')
            weight = 0.
        else:
            weight = float(player_info['WEIGHT'])

        # Get year drafted, or 0 if player was not drafted
        year_drafted = 0 if player_info['DRAFT_YEAR'] == 'Undrafted' else int(player_info['DRAFT_YEAR'])

        return (player_id, player_info['FIRST_NAME'], player_info['LAST_NAME'], height, weight, year_drafted)

    def build(self, reset=True):
        """Build the dataset (to be run only once, or once each season)"""
        # Connect to db
        conn = sqlite.connect(self.db_name)
        cursor = conn.cursor()

        # Build table for first time
        if reset:
            sql_query = """CREATE TABLE player_info (player_id INT, first_name TEXT,
                            last_name TEXT, height REAL, weight REAL, year_drafted INT)"""
            cursor.execute(sql_query)
            conn.commit()

            with open('data/finished.txt', 'w+') as file:
                file.write('0')

        # Get player stats
        all_players = CommonAllPlayers()
        # all_player_info = []
        with open('data/finished.txt', 'r') as file:
            finished = int(file.read())

        player_data = all_players.get_dict()['resultSets'][0]['rowSet']
        pbar = tqdm(total=len(player_data), initial=finished)

        for row in player_data[finished:]:
            try:
                local_player_info = self._individual_player_info(row[0])
            except ValueError as ve:
                print(repr(ve))
                import pdb
                pdb.set_trace()
                continue
            # all_player_info.append(local_player_info)
            finished += 1
            with open('data/finished.txt', 'w+') as file:
                file.write(str(finished))
            pbar.update(1)
            sleep(2)

            # Insert row to table
            sql_query = f"""INSERT INTO player_info VALUES (?, ?, ?, ?, ?, ?)"""
            cursor.execute(sql_query, local_player_info)
            conn.commit()

        pbar.close()

        # # Insert rows to table
        # sql_query = f"""INSERT INTO player_info VALUES (?, ?, ?, ?, ?, ?)"""
        # cursor.executemany(sql_query, all_player_info)
        # conn.commit()

        # Close connection
        conn.close()

if __name__ == '__main__':
    db = PlayerInfoDB(db_name='data/player_info.db')
    db.build(reset=True)
