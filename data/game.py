"""
A class-based way to store all the information for a single game so it can
easily be batched for analysis.

@author: Riley Smith
Created: 5-27-2021
"""
from datetime import timedelta, datetime

from nba_api.stats.endpoints import BoxScoreTraditionalV2 as BoxScore
from nba_api.stats.endpoints import BoxScoreSummaryV2 as BoxScoreSummary
from nba_api.stats.endpoints import PlayerDashboardByLastNGames as LastN
from nba_api.stats.endpoints import CommonPlayerInfo as PlayerInfo

from utils import format_season, nba_dict
import player_info_db

PLAYER_KEYS = [
    'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV',
    'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS'
]

def format_minutes(minutes):
    """Take string format of min:sec and make it float for minutes played"""
    # Return None if this player did not play
    if 'DNP' in minutes:
        return None
    min, sec = minutes.split(':')
    return float(min) + (float(sec) / 60)

def basic_player_info(player_id, current_year, db_name='data/player_info.db'):
    """Get the player's height, weight, and years of experience from the supplied database"""
    # Read info from db
    data = player_info_db.read(player_id, db_name=db_name)
    # Extract relevant info
    try:
        height, weight, year_drafted = data[-3:]
    except:
        return None
    # Get experience
    exp = -1 if year_drafted == 0 else current_year - year_drafted
    undrafted = 1 if year_drafted == 0 else 0
    return height, weight, exp, undrafted

class Game():
    def __init__(self, game_id):
        self.game_id = game_id

        self.game_date = None   # Set by _get_players function
        self.players = self._get_players()

        self.season, self.year = format_season(self.game_date, return_year=True)

        # Get the cumulative stats for each player (this takes a sec)
        self.player_stats = self._cumulative_stats()

    def _get_players(self):
        """
        Get the lists of home and away players (by their IDs) for this game.
        """
        # Fetch the box score
        box = BoxScore(game_id=self.game_id)
        time.sleep(1.8)

        # Get the teams
        box_summary = BoxScoreSummary(game_id=self.game_id)
        time.sleep(1.8)

        series_stats = box_summary.season_series.get_dict()['data'][0]
        home_id = series_stats[1]
        away_id = series_stats[2]

        # Set the game date
        date_string = box_summary.line_score.get_dict()['data'][0][0].split('T')[0]
        self.game_date = datetime.strptime(date_string, '%Y-%m-%d')

        # Get the players, sorted by team
        home_players = []
        away_players = []
        for player_row in box.player_stats.get_dict()['data']:
            # If this player did not play, skip them
            if player_row[9] is None:
                continue
            id = player_row[4]
            minutes = format_minutes(player_row[9])
            # Throw this player out if they did not participate
            if minutes is None:
                continue
            if player_row[1] == home_id:
                home_players.append((id, minutes))
            elif player_row[1] == away_id:
                away_players.append((id, minutes))
            else:
                raise ValueError(f'Player with id {id} has invalid team id: {player_row[1]}.')

        # Sort by minutes played and only take the top 12 players
        home_players = sorted(home_players, key=lambda x: x[1], reverse=True)
        away_players = sorted(away_players, key=lambda x: x[1], reverse=True)

        return {'home': home_players[:10], 'away': away_players[:10]}

    def _cumulative_stats(self):
        """
        Get the cumulative stats for all players in the game up to (but not including)
        this game.
        """
        output_player_stats = {}
        for team in ['home', 'away']:
            team_player_stats = []

            # Get players from dictionary
            players = self.players[team]

            # Get data for each player
            for player_id, _ in tqdm(players):
                # Ping nba api for LastN stats
                last_date = datetime.strftime(self.game_date - timedelta(days=1), "%m-%d-%Y")
                player_stats = LastN(player_id=player_id, season=self.season, date_to_nullable=last_date)
                # Get this player's height, weight, experience
                height, weight, experience, undrafted = basic_player_info(player_id, self.year)
                # Get total, last 5, last 15
                try:
                    total_stats = nba_dict(player_stats.overall_player_dashboard.get_dict())
                except IndexError as ie:
                    # This player has no data yet on the season, so skip them
                    continue   # TODO: Grab last season's stats instead
                try:
                    last_5 = nba_dict(player_stats.last5_player_dashboard.get_dict())
                except IndexError as ie:
                    # Occurs when 5 games haven't been completed yet. In this case, use total stats
                    last_5 = total_stats
                try:
                    last_15 = nba_dict(player_stats.last15_player_dashboard.get_dict())
                except IndexError as ie:
                    last_15 = total_stats
                # Pull out weighted average of player usage
                player_usage = (0.5 * last_5['MIN']) + (0.25 * last_15['MIN']) + (0.25 * total_stats['MIN'])
                # Get just the keys that you are interested in
                total_stats = [total_stats[k] for k in PLAYER_KEYS]
                last_5 = [last_5[k] for k in PLAYER_KEYS]
                last_15 = [last_15[k] for k in PLAYER_KEYS]
                # Concatenate them all to make this player's singular datapoint
                output_stats = []
                for stats in zip(total_stats, last_5, last_15):
                    output_stats += stats
                # Add height, weight, experience
                output_stats += [height, weight, experience]
                # Append it to list of player stats
                team_player_stats.append((player_usage, output_stats))
                time.sleep(1.8)
            # Add player stats to the overall output dictionary
            output_player_stats[team] = team_player_stats
        return output_player_stats
