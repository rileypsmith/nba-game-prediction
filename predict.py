"""
A script for running predictions.

@author: Riley Smith
Created: 11-7-2021
"""
import argparse
from datetime import datetime, timedelta
import joblib

from data.collect_data import list_available_games, format_game_data

def main(game_date=None):
    """
    Make predictions for each game on a given date and print them nicely in
    the console. Currently setup to use random forest classifier for predicitons.
     This requires that you have a trained model saved at
    `models/random_forest.joblib`.
    """
    # Load the model
    clf = joblib.load('models/random_forest.joblib')

    # Get list of available games
    if game_date is None:
        game_date = datetime.today()
    print(f'Making predictions for {game_date}:\n')
    available_games = list_available_games(game_date)

    # For each game, get the data, formatted as input to the model
    game_data = []
    for game in available_games:
        data = format_game_data(game['home'], game['away'], game_date)
        game['data'] = data
        game_data.append(game)
        print()

    # Now make a prediction for each one and print it to the console
    for game in game_data:
        pred = clf.predict_proba(game['data']).squeeze()[0]
        # print('red is: ', pred)
        predicted_winner = game['home'] if pred > 0.5 else game['away']
        win_prob = round(pred * 100, 2) if pred > 0.5 else round((1 - pred) * 100, 2)
        print(f'{game["away"]} at {game["home"]}')
        print(f'\t{predicted_winner} predicted to win with {win_prob}% chance of winning')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line arguments for NBA game prediction.')
    parser.add_argument('--date', type=str, default='today',
                        help='The date to make predicitons for. Could also be "today" or "tomorrow".')
    args = parser.parse_args()

    if args.date == 'today':
        game_date = datetime.today()
    elif args.date == 'tomorrow':
        game_date = datetime.today() + timedelta(days=1)
    else:
        game_date = args.date

    main(game_date=game_date)
