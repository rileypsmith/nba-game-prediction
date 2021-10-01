"""
Class-based collection of all data for a team (their players and relevant team
stats). Contains methods for batching the data for analysis.

UPDATE 9-30-21: This is deprecated.

@author: Riley Smith
Created: 5-28-2021
"""
from nba_api.stats.endpoints import CommonTeamRoster

class Team():
    def __init__(team_id, date_to):
        self.id = team_id
        self.date_to = date_to
        self.season = format_season(date_to)

        # Get player info
        self.player_data = self._get_player_data()

    def _get_player_data(self):
        # Get the actual player IDs from the roster
        roster = CommonTeamRoster(team_id=self.id, season=self.season).common_team_roster.get_dict()
