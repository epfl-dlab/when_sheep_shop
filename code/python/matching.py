#! /usr/bin/env python
# coding=utf-8
#
# Copyright © 2017 Gael Lederrey <gael.lederrey@epfl.ch>
#
# Distributed under terms of the MIT license

from classes.helpers import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
import copy
import json
import os


class Matching:
    def __init__(self):
        """
        Initialize the class
        """
        self.data_folder = '../data/'
        self.breweries = {}
        self.beers = {}
        self.users = {}

        self.thresholds = {'breweries': {'sim': 0.8, 'diff': 0.3},
                           'beers': {'sim': 0.8, 'diff': 0.4},
                           'users': {'sim': 0.9}}

        self.types = {
            'ba': {
                'beer_name': str,
                'beer_id': int,
                'brewery_name': str,
                'brewery_id': int,
                'style': str,
                'abv': float,
                'date': int,
                'user_name': str,
                'user_id': str,
                'appearance': float,
                'aroma': float,
                'palate': float,
                'taste': float,
                'overall': float,
                'rating': float,
                'text': str,
                'review': bool
            },
            'rb': {
                'beer_name': str,
                'beer_id': int,
                'brewery_name': str,
                'brewery_id': int,
                'style': str,
                'abv': float,
                'date': int,
                'user_name': str,
                'user_id': int,
                'appearance': float,
                'aroma': float,
                'palate': float,
                'taste': float,
                'overall': float,
                'rating': float,
                'text': str
            }
        }

        if not os.path.exists(self.data_folder + 'matched'):
            os.makedirs(self.data_folder + 'matched')

        if not os.path.exists(self.data_folder + 'tmp'):
            os.makedirs(self.data_folder + 'tmp')

    def match_breweries(self):
        """
        Match the breweries and create a df with the matched breweries
        """

        print('Matching the breweries')

        # Load the df of the breweries
        self.breweries['ba'] = pd.read_csv(self.data_folder + 'ba/breweries.csv')
        self.breweries['rb'] = pd.read_csv(self.data_folder + 'rb/breweries.csv')

        # Get the smaller and bigger df
        if len(self.breweries['ba']) < len(self.breweries['rb']):
            small = 'ba'
            big = 'rb'
        else:
            small = 'rb'
            big = 'ba'

        # Create the big and small df
        small_df = self.breweries[small].drop_duplicates(['name'], keep=False)
        small_df.index = range(len(small_df))
        big_df = self.breweries[big].drop_duplicates(['name'], keep=False)
        big_df.index = range(len(big_df))

        # Prepare corpus
        corpus = list(set(small_df['name'].unique()) | set(big_df['name'].unique()))

        # Use the vectorizer to get the vocabulary
        tfidf_vect = TfidfVectorizer()
        tfidf_vect.fit_transform(corpus)
        vocabulary = tfidf_vect.get_feature_names()

        # Prepare the corpus with the big df
        corpus = list(big_df['name'])

        # Vectorize and train
        tfidf_vect = TfidfVectorizer(vocabulary=vocabulary)
        tfidf_train = tfidf_vect.fit_transform(corpus)

        # Test with the small df
        tfidf_test = tfidf_vect.transform(small_df['name'])

        # Get the cosine similarity matrix
        sim = cosine_similarity(tfidf_test, tfidf_train)

        # Number of best matches we want
        nbr_matches = 2

        # Prepare the JSON df
        df_json = {'ba': {'id': [], 'location': [], 'name': [], 'nbr_beers': []},
                   'rb': {'id': [], 'location': [], 'name': [], 'nbr_beers': []},
                   'scores': {'sim': [], 'diff': []}}

        # Go through all breweries in the small df
        for idx, str_ in enumerate(small_df['name']):
            sim_str = sim[idx]

            result_str = []

            sim_idx = (-sim_str).argsort()[:nbr_matches]

            # Get the two best matches
            for i in range(nbr_matches):
                result_str.append({'name': big_df['name'][sim_idx[i]], 'score': sim_str[sim_idx[i]]})

            # Check for the similarity
            score_big_enough = result_str[0]['score'] > self.thresholds['breweries']['sim']

            # Check for the difference
            delta = result_str[0]['score'] - result_str[1]['score']
            diff_big_enough = delta > self.thresholds['breweries']['diff']

            # Get location of small df
            loc_small = small_df.ix[idx]['location']
            # Get location of big df
            subdf = big_df[big_df['name'] == result_str[0]['name']]
            idx_big = subdf.index[0]
            loc_big = subdf['location'].ix[idx_big]

            # Check for same location (exact match)
            same_location = (loc_small == loc_big)

            if score_big_enough and diff_big_enough and same_location:
                # We have a match! Add it to the JSON

                for key in df_json['ba'].keys():
                    df_json[small][key].append(small_df.ix[idx][key])

                for key in df_json['ba'].keys():
                    df_json[big][key].append(big_df.ix[idx_big][key])

                df_json['scores']['sim'].append(result_str[0]['score'])
                df_json['scores']['diff'].append(delta)

        # Create the pandas DF from the dict
        df = pd.DataFrame.from_dict({(i, j): df_json[i][j]
                                     for i in df_json.keys()
                                     for j in df_json[i].keys()})

        # Save it
        df.to_csv(self.data_folder + 'matched/breweries.csv', index=False)

    def match_beers(self):
        """
        Match the beers and create a df with the matched beers
        """

        print('Matching the beers')

        # Load the DF of matched breweries
        matched_breweries = pd.read_csv(self.data_folder + 'matched/breweries.csv', header=[0, 1])

        # Add the beers to this class
        self.beers['ba'] = pd.read_csv(self.data_folder + 'ba/beers.csv')
        self.beers['rb'] = pd.read_csv(self.data_folder + 'rb/beers.csv')

        # Compute the column of beer names without the brewery name
        for key in ['ba', 'rb']:
            beer_wout_brewery_list = []
            for i in self.beers[key].index:
                row = self.beers[key].ix[i]
                beer_name = row['beer_name']
                beer_name = ''.join(l for l in beer_name if l not in string.punctuation)

                brewery_name = row['brewery_name']
                brewery_name = ''.join(l for l in brewery_name if l not in string.punctuation)

                set_diff = set(beer_name.split(' ')) - set(brewery_name.split(' '))
                set_diff.discard('')

                beer_wout_brewery = ' '.join(set_diff)

                # The name of the beer is composed of words in the brewery name
                if beer_wout_brewery == '':
                    beer_wout_brewery = ' '.join(set(beer_name.split(' ')))

                # The name of the beer is composed of punctuations
                if beer_wout_brewery == '':
                    beer_wout_brewery = row['beer_name']

                beer_wout_brewery_list.append(beer_wout_brewery)
            self.beers[key].loc[:, 'beer_wout_brewery_name'] = beer_wout_brewery_list

        # Prepare the Corpus to get the vocabulary for the TF-IDF vectorizer
        corpus = list(set(self.beers['ba']['beer_wout_brewery_name']) | set(self.beers['rb']['beer_wout_brewery_name']))

        # Vectorize and get the vocabulary
        tfidf_vect = TfidfVectorizer()
        tfidf_vect.fit_transform(corpus)
        vocabulary = tfidf_vect.get_feature_names()

        # Prepare the DF JSON
        df_json = {'ba': {}, 'rb': {}, 'scores': {'sim': [], 'diff': []}}
        for key in ['ba', 'rb']:
            cols = list(self.beers[key].columns)
            for col in cols:
                df_json[key][col] = []

        # Go through all matched breweries
        for i in matched_breweries.index:
            row = matched_breweries.ix[i]

            # Create subset of beers from hte matched breweries
            subset = {}
            for key in ['ba', 'rb']:
                subset[key] = self.beers[key][self.beers[key]['brewery_id'] == row[key]['id']]
                subset[key].index = range(len(subset[key]))

            # Get small and big subsets
            if len(subset['ba']) < len(subset['rb']):
                small = 'ba'
                big = 'rb'
            else:
                small = 'rb'
                big = 'ba'

            if len(subset[small]) > 0:
                # Train the TF-IDF Vectorizer
                tfidf_vect = TfidfVectorizer(vocabulary=vocabulary)
                tfidf_train = tfidf_vect.fit_transform(subset[big]['beer_wout_brewery_name'])

                # And test it
                tfidf_test = tfidf_vect.transform(subset[small]['beer_wout_brewery_name'])

                # Get matrix of cosine similarity
                sim = cosine_similarity(tfidf_test, tfidf_train)

                nbr_matches = 2

                # Go through all the beers of the matched breweries
                for j in subset[small].index:
                    # Get the row of the small subset
                    row_small = subset[small].ix[j]

                    # Get the similarities with this beer
                    sim_str = sim[j]

                    sim_idx = (-sim_str).argsort()[:nbr_matches]

                    # Get the best match
                    row_big = subset[big].ix[sim_idx[0]]

                    # Best score
                    score = sim_str[sim_idx[0]]

                    # And difference of score between 1st and 2nd match
                    if len(sim_idx) > 1:
                        delta = sim_str[sim_idx[0]] - sim_str[sim_idx[1]]
                    else:
                        delta = sim_str[sim_idx[0]]

                    # Check for the similarity
                    score_big_enough = score > self.thresholds['beers']['sim']

                    # Check for the difference
                    diff_big_enough = delta > self.thresholds['beers']['diff']

                    # Same ABV
                    same_abv = (row_small['abv'] == row_big['abv'])

                    if score_big_enough and diff_big_enough and same_abv:
                        # We have a match! Add it to the JSON

                        # Add row of small subset to df_json
                        for col in subset[small]:
                            df_json[small][col].append(row_small[col])

                        # Add row of big subset to df_json
                        for col in subset[big]:
                            df_json[big][col].append(row_big[col])

                        # Add score and delta
                        df_json['scores']['sim'].append(score)
                        df_json['scores']['diff'].append(delta)

        # Create the pandas DF from the dict
        df = pd.DataFrame.from_dict({(i, j): df_json[i][j]
                                     for i in df_json.keys()
                                     for j in df_json[i].keys()})

        # Remove duplicates
        df = df[~df['ba']['beer_id'].duplicated(keep=False)]
        df = df[~df['rb']['beer_id'].duplicated(keep=False)]
        df.index = range(len(df))

        # Save it
        df.to_csv(self.data_folder + 'matched/beers.csv', index=False)

    def match_users_exact(self):
        """
        Match the users and create a df with the matched users

        Matching is done with an exact match
        """

        print('Matching the users')

        # Add the users to the matcher
        self.users['ba'] = pd.read_csv(self.data_folder + 'ba/users.csv')
        self.users['rb'] = pd.read_csv(self.data_folder + 'rb/users.csv')

        # Compute the lowercase letter of usernames
        for key in ['ba', 'rb']:
            low = [x.lower() for x in self.users[key]['user_name']]
            self.users[key].loc[:, 'user_name_lower'] = low

        # Get the small and big df
        if len(self.users['ba']) < len(self.users['rb']):
            small = 'ba'
            big = 'rb'
        else:
            small = 'rb'
            big = 'ba'

        # Prepare the JSON DF
        df_json = {}
        for key in ['ba', 'rb']:
            df_json[key] = {}
            for col in list(self.users[key].columns):
                df_json[key][col] = []

        # Go through all users in the small df
        for i in self.users[small].index:
            row_small = self.users[small].ix[i]

            # Get the user who match on the username and location
            row_big = self.users[big][(self.users[big]['user_name_lower'] == row_small['user_name_lower']) &
                                         (self.users[big]['location'] == row_small['location'])]

            # Add it to the DF JSON
            if len(row_big) > 0:
                idx = row_big.index[0]
                row_big = row_big.ix[idx]

                # Fill from the small
                for key in df_json[small]:
                    df_json[small][key].append(row_small[key])

                # Fill from the big
                for key in df_json[big]:
                    df_json[big][key].append(row_big[key])

        # Create DataFrame pandas DF from the dict
        df = pd.DataFrame.from_dict({(i, j): df_json[i][j]
                                     for i in df_json.keys()
                                     for j in df_json[i].keys()})

        # Save it
        df.to_csv(self.data_folder + 'matched/users.csv', index=False)

    def match_users_approx(self):
        """
        Match the users and create a df with the matched users

        Matching is done with TF-IDF vectorizer + cosine similarity
        Users are matched if username is close enough and same location
        """

        print('Matching the users')

        # Add the users to the matcher
        self.users['ba'] = pd.read_csv(self.data_folder + 'ba/users.csv')
        self.users['rb'] = pd.read_csv(self.data_folder + 'rb/users.csv')

        # Compute the lowercase letter of usernames
        for key in ['ba', 'rb']:
            low = [x.lower() for x in self.users[key]['user_name']]
            self.users[key].loc[:, 'user_name_lower'] = low

        # Get the small and big df
        if len(self.users['ba']) < len(self.users['rb']):
            small = 'ba'
            big = 'rb'
        else:
            small = 'rb'
            big = 'ba'

        corpus = list(set(self.users['ba']['user_name_lower']) | set(self.users['rb']['user_name_lower']))

        # Vectorize and get the vocabulary
        tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
        tfidf_vect.fit_transform(corpus)
        vocabulary = tfidf_vect.get_feature_names()

        # Prepare the JSON DF
        df_json = {}
        for key1 in ['ba', 'rb']:
            df_json[key] = {}
            for col in list(self.users[key].columns):
                df_json[key][col] = []

        df_json['scores'] = {'sim': []}

        for i in self.users[small].index:
            row_small = self.users[small].ix[i]

            subset_big = self.users[big][self.users[big]['location'] == row_small['location']]
            subset_big.index = range(len(subset_big))

            if len(subset_big) > 0:

                # Train the TF-IDF Vectorizer
                tfidf_vect = TfidfVectorizer(vocabulary=vocabulary, analyzer='char', ngram_range=(2, 2))
                tfidf_train = tfidf_vect.fit_transform([row_small['user_name_lower']])

                # And test it
                tfidf_test = tfidf_vect.transform(subset_big['user_name_lower'])

                # Get matrix of cosine similarity
                sim = cosine_similarity(tfidf_test, tfidf_train)
                sim = sim[:, 0]

                nbr_matches = 1

                sim_idx = (-sim).argsort()[:nbr_matches]

                score = sim[sim_idx[0]]

                row_big = subset_big.ix[sim_idx[0]]

                if score >= 0.8:
                    # Add small
                    for col in self.users[small].columns:
                        df_json[small][col].append(row_small[col])

                    # Add big
                    for col in self.users[big].columns:
                        df_json[big][col].append(row_big[col])

                    df_json['scores']['sim'].append(score)

        # Create the pandas DF from the dict
        df = pd.DataFrame.from_dict({(i, j): df_json[i][j]
                                     for i in df_json.keys()
                                     for j in df_json[i].keys()})

        # Save it
        df.to_csv(self.data_folder + 'matched/users_approx.csv', index=False)

    def parse_ratings_for_pairs_users_beers(self):
        """
        Parse the ratings to get all the pairs users - beers
        """

        # Parse the ratings to get all the pairs users - beers
        users_beers = {}

        # Go through both data sets
        for key in ['ba', 'rb']:
            print('Parse ratings from {}'.format(key.upper()))

            # Get the generator
            gen = parse('../data/{}/ratings.txt.gz'.format(key))

            users_beers[key] = {}

            # Go through all the ratings
            for item in gen:
                usr = item['user_id']
                beer = item['beer_id']

                if usr not in users_beers[key].keys():
                    users_beers[key][usr] = []

                # Add it to the Dict
                users_beers[key][usr].append(beer)

        with open(self.data_folder + 'tmp/users_beers.json', 'w') as file:
            json.dump(users_beers, file)

    def match_ratings_first_part(self):
        """
        Before running this function, you need to run the function 'parse_ratings_for_pairs_users_beers'.

        This function creates a pandas DF with all the pairs users - beers with same user and same beer on
        both website.

        To get the final DF, you need to run the file called parse_ratings_for_users_beers.py and finally
        run the function match_ratings_second_part.
        """

        # Open all the pairs
        with open('../data/tmp/users_beers.json') as file:
            users_beers = json.load(file)

        # Load the DF with all the matched users
        df_users = pd.read_csv(self.data_folder + 'matched/users.csv', header=[0, 1])

        # Load the DF with all the matched beers
        df_beers = pd.read_csv(self.data_folder + 'matched/beers.csv', header=[0, 1])

        # Get all the beers of the matched users (stripping down the dict users_beers)
        matched_users_beers = {}

        # Go through both data sets
        for key in ['rb', 'ba']:
            matched_users_beers[key] = {}

            # Get the list of users for a given data set
            list_usr_id = [str(it) for it in list(df_users[key]['user_id'])]

            # Take all these users from the dict users_beers
            for usr in list_usr_id:
                matched_users_beers[key][usr] = copy.deepcopy(users_beers[key][usr])

        # Get all the matched beers with the matched users (stripping down the dict matched_users_beers)
        matched_users_matched_beers = {}

        # Go through both data sets
        for key in ['rb', 'ba']:
            matched_users_matched_beers[key] = {}

            # Get the set of matched beers
            set_matched_beers = set([str(b) for b in list(df_beers[key]['beer_id'])])

            # Go through all the matched users
            for usr in matched_users_beers[key].keys():
                # Take a user's set of beer he rated on a website
                set_beers = set(matched_users_beers[key][usr])

                # and keep only the beers that matched
                matching = list(set_beers & set_matched_beers)
                if len(matching) > 0:
                    matched_users_matched_beers[key][usr] = matching

        # Now, we have a dict with all the matched users and all the subsets of beers that matched he rated
        # for both data sets. The goal, now, is to link the users and beers between the two data sets.

        # Create an empty json dict
        df_json = {}
        for key in ['ba', 'rb']:
            df_json[key] = {'user_id': [], 'beer_id': []}

        # Go through all users who matched
        for i in df_users.index:
            row = df_users.ix[i]

            # We check if the user is present in the dict matched_users_matched_beers, i.e. if he has rated a beer that
            # matched between the two websites
            users_present = True
            for key in ['ba', 'rb']:
                if str(row[key]['user_id']) not in matched_users_matched_beers[key].keys():
                    users_present = False

            # If he's present, then we can continue
            if users_present:

                # We take all the beers he rated between the two websites
                beers = {}
                for key in ['ba', 'rb']:
                    beers[key] = matched_users_matched_beers[key][str(row[key]['user_id'])]

                # Find the smallest (and biggest) subset between both websites
                if len(beers['ba']) < len(beers['rb']):
                    small = 'ba'
                    big = 'rb'
                else:
                    small = 'rb'
                    big = 'ba'

                # Go through all elements in the smallest subset of matched beers
                for elem in beers[small]:

                    # We get the beer in the biggest subset (in the other data set) that matched with the current beer.
                    row_beer = df_beers[df_beers[small]['beer_id'] == int(elem)]
                    row_beer = row_beer.ix[row_beer.index[0]]

                    # We check if this beer is in the subset of matched beers this user rated
                    if str(row_beer[big]['beer_id']) in beers[big]:
                        # If it's the case, we fill the dict

                        # Fill the small
                        df_json[small]['user_id'].append(row[small]['user_id'])
                        df_json[small]['beer_id'].append(int(elem))

                        # Fill the big
                        df_json[big]['user_id'].append(row[big]['user_id'])
                        df_json[big]['beer_id'].append(int(row_beer[big]['beer_id']))

        # Once the dict is finished, we create the pandas DF
        df = pd.DataFrame.from_dict({(i, j): df_json[i][j]
                                     for i in df_json.keys()
                                     for j in df_json[i].keys()})

        # And save it
        df.to_csv(self.data_folder + 'tmp/same_user_same_beer.csv', index=False)

    def match_ratings_second_part(self):
        """
        Before running this function, you need to run the code parse_ratings_for_users_beers.py

        With the previous code, we have two dicts (one for each data set) of the ratings from the matched beers
        and matched users, i.e. the matched reviews. This part is only about transforming the dict into a pandas DF.
        """

        # First, we load the two dict with all the ratings
        data = {}
        for key in ['rb', 'ba']:
            with open(self.data_folder + 'tmp/users_beers_{}.json'.format(key)) as file:
                data[key] = json.load(file)

        # We also load the pandas DF with the pairs of matched users - matched beers
        df = pd.read_csv(self.data_folder + 'tmp/same_user_same_beer.csv', header=[0, 1])

        # Create an empty dict with all the elements in the ratings
        json_reviews = {}
        for key1 in self.types.keys():
            tmp = {}
            for key2 in self.types[key1].keys():
                tmp[key2] = []
            json_reviews[key1] = copy.deepcopy(tmp)

        # Then we go through all the pairs in the df and add the full ratings corresponding to these pairs to the dict
        for i in df.index:
            row = df.ix[i]
            for key in ['ba', 'rb']:
                # Get user_id and beer_id (to get the full rating in the dict data)
                user_id = str(row[key]['user_id'])
                beer_id = str(row[key]['beer_id'])

                # Get the reviews
                rev = data[key][user_id][beer_id]

                # Add all the elements in the reviews in the final dict with their specific type
                for elem in rev.keys():
                    json_reviews[key][elem].append(self.types[key][elem](rev[elem]))

        # Once the dict is finished, we create the pandas DF
        df_reviews = pd.DataFrame.from_dict({(i, j): json_reviews[i][j]
                                             for i in json_reviews.keys()
                                             for j in json_reviews[i].keys()})

        # And save it
        df_reviews.to_csv(self.data_folder + 'matched/ratings.csv', index=False)
