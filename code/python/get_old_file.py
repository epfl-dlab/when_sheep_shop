#! /usr/bin/env python
# coding=utf-8
#
# Copyright © 2017 Gael Lederrey <gael.lederrey@epfl.ch>
#
# Distributed under terms of the MIT license.

import ast
import gzip
import json
import numpy as np
import pandas as pd
from python.helpers import parse, flatten
from datetime import datetime

"""
This file is used to compute the old file that was used
to create Table 5. You don't have to use this to get the
results of Table 5 with the correct file.
"""

def run():

    # Some parameters
    data_folder = '../data/'
    notext = {'ba': 'nan', 'rb': ''}
    min_nbr_rats = 5

    """ Create txt.gz files with ratings (with text) of matched beers """
    # Load DF of matched beers
    df = pd.read_csv(data_folder + 'matched/beers.csv', header=[0,1])
    # Get the ids
    beer_ids = {'ba': np.array(df['ba']['beer_id']), 'rb': np.array(df['rb']['beer_id'])}

    # Go through BA and RB
    for key in ['ba', 'rb']:
        print('Parse {} ratings'.format(key.upper()))

        # Get iterator
        gen = parse(data_folder + key + '/ratings.txt.gz'.format(key))

        # Open the new gzip file
        file_ = gzip.open(data_folder + 'matched/ratings_with_text_{}.txt.gz'.format(key), 'wb')

        # Go through iterator
        for item in gen:
            text = item['text']
            # Check if the beer corresponds to a matched beers
            if text != notext[key] and int(item['beer_id']) in beer_ids[key]:
                # If yes, add it to the new gzip file
                for key_dct in item.keys():
                    file_.write('{}: {}\n'.format(key_dct, item[key_dct]).encode('utf-8'))
                file_.write('\n'.encode('utf-8'))

        file_.close()

    """ Parse the ratings with text from the matched beers to put it in a JSON file """

    # Get the ratings for every year for both BA and RB
    ratings_year = {'ba': {}, 'rb': {}}

    # Go through RB and BA
    for key in ratings_year.keys():
        print('Parsing {} reviews.'.format(key.upper()))

        # Get the iterator
        gen = parse(data_folder + '/matched/ratings_with_text_{}.txt.gz'.format(key))

        # Go through the iterator
        for item in gen:

            # Get the data and then the year
            date = int(item['date'])
            year = datetime.fromtimestamp(date).year

            if year not in ratings_year[key].keys():
                ratings_year[key][year] = []

            # Add the rating in the correct year
            ratings_year[key][year].append(float(item['rating']))

    with open(data_folder + 'tmp/rating_per_year_matched_with_text.json', 'w') as outfile:
        json.dump(ratings_year, outfile)

    """ Compute the parameters for the zscore of this subset of ratings """

    # Get the file rating_per_year from notebook 3
    with open(data_folder + 'tmp/rating_per_year_matched_with_text.json', 'r') as infile:
        ratings_year = json.load(infile)

    # Prepare the dict of mean and std per year to compute the zscore
    z_score_params = {}

    # Go through BA and RB
    for key in ratings_year.keys():
        z_score_params[key] = {}
        # Go through each year
        for y in ratings_year[key].keys():
            # Add the average and STD
            z_score_params[key][y] = {'mean': np.mean(ratings_year[key][y]), 'std': np.std(ratings_year[key][y])}
            if z_score_params[key][y]['std'] == 0:
                z_score_params[key][y]['std'] = 1

    # Fill some missing years
    z_score_params['ba'][1996] = {'mean': 0, 'std': 1}
    z_score_params['ba'][1997] = {'mean': 0, 'std': 1}

    # And save the file
    with open(data_folder + 'tmp/z_score_params_matched_ratings_with_text.json', 'w') as file:
            json.dump(z_score_params, file)

    """ Compute all the ratings per id """

    # get the ratings for these beers
    ratings = {'ba': {}, 'rb': {}}

    # Go through BA and RB
    for key in ratings.keys():
        print('Parse {} ratings'.format(key.upper()))
        # get the iterator
        gen = parse(data_folder + 'matched/ratings_with_text_{}.txt.gz'.format(key))

        # Go through the iterator
        for item in gen:

            # Get the beer_id, the rating and the date
            beer_id = item['beer_id']
            rating = item['rating']
            date = item['date']

            if beer_id not in ratings[key].keys():
                ratings[key][beer_id] = {'date': [], 'rating': []}

            # And add them
            ratings[key][beer_id]['date'].append(int(date))
            ratings[key][beer_id]['rating'].append(float(rating))

    """ Compute the global averages and prepare to compute the time series """

    # Compute the global averages
    global_average = {'ba': {'rating': 0, 'z_score': 0, 'std': 0},
                      'rb': {'rating': 0, 'z_score': 0, 'std': 0}}

    with open(data_folder + 'tmp/z_score_params_matched_ratings_with_text.json') as file:
        z = json.load(file)

    # Go through BA and RB
    for key in ratings.keys():
        all_ratings = []
        all_z_score = []
        nbr = 0

        # Go through all ids
        for id_ in ratings[key].keys():
            rats = ratings[key][id_]['rating']
            dates = ratings[key][id_]['date']

            # Get year and compute zscores
            years = [str(datetime.fromtimestamp(d).year) for d in dates]
            z_scores = [(r-z[key][y]['mean'])/z[key][y]['std'] for r,y in zip(rats, years)]

            # Add the ratings and zscores to the global array
            all_ratings.append(rats)
            all_z_score.append(z_scores)

        # Flatten the array
        all_ratings = flatten(all_ratings)
        all_z_score = flatten(all_z_score)

        # Compute the global averages and std
        global_average[key]['std'] = np.std(all_ratings)
        global_average[key]['rating'] = np.mean(all_ratings)
        global_average[key]['z_score'] = np.mean(all_z_score)

    """ Compute the time series """

    # Get the matched beers
    beers = pd.read_csv(data_folder + 'matched/beers.csv', header=[0,1])

    # Get the matched beers with at least 5 ratings
    beers = beers[(beers['ba']['nbr_ratings'] >= min_nbr_rats) & (beers['rb']['nbr_ratings'] >= min_nbr_rats)]
    beers.index = range(len(beers))

    # Create the dict
    df_json = {'ba': {'beer_id': [], 'dates': [], 'nbr_ratings': [], 'ratings': [], 'z_scores': [], 'avg_ratings': [], 'avg_z_scores': []},
               'rb': {'beer_id': [], 'dates': [], 'nbr_ratings': [], 'ratings': [], 'z_scores': [], 'avg_ratings': [], 'avg_z_scores': []}}

    # Go through all matched beers
    for i in beers.index:
        row = beers.iloc[i]

        ba_id = row['ba']['beer_id']
        rb_id = row['rb']['beer_id']

        if str(ba_id) in ratings['ba'].keys() and str(rb_id) in ratings['rb'].keys():

            # Go through BA and RB
            for key in ['ba', 'rb']:
                # Add the beer_id
                df_json[key]['beer_id'].append(row[key]['beer_id'])

                # get the ratings
                ratings_user = ratings[key][str(row[key]['beer_id'])]

                # Inverse the date and ratings
                dates = ratings_user['date'][::-1]
                rats = ratings_user['rating'][::-1]

                # Compute zscore
                years = [str(datetime.fromtimestamp(d).year) for d in dates]
                z_scores = [(r-z[key][y]['mean'])/z[key][y]['std'] for r,y in zip(rats, years)]

                # Transform list into np.array
                dates = np.array(dates)
                rats = np.array(rats)
                z_scores = np.array(z_scores)

                # Make sure everything is sorted
                idx = np.argsort(dates)
                dates = dates[idx]
                rats = rats[idx]
                z_scores = z_scores[idx]

                # Add to the dict
                df_json[key]['dates'].append(list(dates))
                df_json[key]['ratings'].append(list(rats))
                df_json[key]['nbr_ratings'].append(len(rats))
                df_json[key]['z_scores'].append(list(z_scores))
                df_json[key]['avg_ratings'].append(np.mean(rats))
                df_json[key]['avg_z_scores'].append(np.mean(z_scores))

    # Transform dict into DF
    df = pd.DataFrame.from_dict({(i, j): df_json[i][j]
                                 for i in df_json.keys()
                                 for j in df_json[i].keys()})

    # Save the DF
    df.to_csv(data_folder + 'tmp/time_series_with_text.csv', index=False)
