import json
import pandas as pd
import gzip


def parse(filename):
    """
    Parse a txt.gz file and return a generator for it
    Copyright © 2017 Gael Lederrey <gael.lederrey@epfl.ch>
    :param filename: name of the file
    :return: Generator to go through the file
    """
    file = gzip.open(filename, 'rb')
    entry = {}
    # Go through all the lines
    for line in file:
        # Transform the string-bytes into a string
        line = line.decode("utf-8").strip()

        # We check for a colon in each line
        colon_pos = line.find(":")
        if colon_pos == -1:
            # if no, we yield the entry
            yield entry
            entry = {}
            continue
        # otherwise, we add the key-value pair to the entry
        key = line[:colon_pos]
        value = line[colon_pos + 2:]
        entry[key] = value


def parse_ratings_for_users_beers():
    df = pd.read_csv('../../data/tmp/same_user_same_beer.csv', header=[0, 1])
    for key in ['ba', 'rb']:
        print('Parse ratings from {}'.format(key.upper()))

        json_dic = {}

        gen = parse('../../data/{}/ratings.txt.gz'.format(key))

        for item in gen:

            user_id = item['user_id']
            if key == 'rb':
                user_id = int(user_id)
            beer_id = int(item['beer_id'])

            a = df[(df[key]['user_id'] == user_id) & (df[key]['beer_id'] == beer_id)]
            if len(a) > 0:
                if user_id not in json_dic.keys():
                    json_dic[user_id] = {}

                json_dic[user_id][beer_id] = item

        with open('../../data/tmp/users_beers_{}.json'.format(key), 'w') as file:
            json.dump(json_dic, file)

if __name__ == '__main__':
    parse_ratings_for_users_beers()
