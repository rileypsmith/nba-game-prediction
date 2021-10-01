"""
Find games that have some issue with them. Some games after running through
initial processing had missing data, so we just want to find those.

@author: Riley Smith
Created: ??? (didn't write it down at the time)
"""
import h5py

with h5py.File('data/games.hdf5', 'a') as file:
    seasons = file.keys()
    problem_dict = {}
    for season in seasons:
        problem_dict[season] = []
        for game in file[season]:
            if game == 'finished':
                continue
            home_data = file[season][game]['away'][:]
            if not (len(home_data) > 0):
                # del file[season][game]
                problem_dict[season].append(game)

# Use Python debugger to examine problematic games
import pdb
pdb.set_trace()
