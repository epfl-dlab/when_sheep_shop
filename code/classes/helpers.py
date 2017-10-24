#! /usr/bin/env python
# coding=utf-8
#
# Copyright © 2017 Gael Lederrey <gael.lederrey@epfl.ch>
#
# Distributed under terms of the MIT license.

import multiprocessing as mp
import numpy as np
import gzip


def old_parse(filename):
    """
    Return a Generator from a txt.gz file.
    Originally written by Julian McAuley @ UCSD
    Updated by Gael Lederrey @ EPFL
    :param filename: name of the file
    :return: Generator to go through the file
    """
    f = gzip.open(filename, 'rb')
    entry = {}
    for l in f:
        try:
            l = l.decode("utf-8").strip()
        except UnicodeDecodeError:
            try:
                l = l.decode("ISO-8859-1").strip()
            except UnicodeDecodeError:
                raise ValueError(l, " cannot be decoded")
        colon_pos = l.find(":")
        if colon_pos == -1:
            yield entry
            entry = {}
            continue
        e_name = l[:colon_pos]
        rest = l[colon_pos + 2:]
        entry[e_name] = rest


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


def bootstrap(nbr_draws, size_array, nbr_threads=None):
    if nbr_threads is None:
        nbr_threads = mp.cpu_count()

    pool = mp.Pool(processes=nbr_threads)

    bootstraps = []

    for i in range(nbr_draws):
        distrib = pool.apply_async(np.random.randint, args=(0, size_array, size_array))

        bootstraps.append(distrib.get())

    pool.close()
    pool.join()

    return bootstraps


def bayesian(avg, nbr_ratings):
    """
    Bayesian Formula for RateBeer ratings

    :param avg: Current average
    :param nbr_ratings: Number of ratings
    :return: Bayesian value
    """
    # Values found with optimization on Pearson's r
    min_votes = 3.94498112
    midpoint = 2.84003453

    return (nbr_ratings/(nbr_ratings + min_votes)) * avg + (min_votes/(nbr_ratings + min_votes)) * midpoint


def normalize_rb(rating, aspect):
    """
    Normalize the RB ratings between 0 and 5

    :param rating: value of the rating
    :param aspect: name of the aspect
    :return: normalized rating
    """

    if aspect == 'overall':
        return rating/4
    elif aspect == 'aroma' or aspect == 'taste':
        return rating/2
    else:
        return rating


def flatten(l):
    """
    Flatten a list of lists
    :param l: List of lists
    :return: flattened list
    """
    try:
        return [item for sublist in l for item in sublist]
    except TypeError:
        return l