#! /usr/bin/env python
# coding=utf-8
#
# Copyright © 2017 Gael Lederrey <gael.lederrey@epfl.ch>
#
# Distributed under terms of the MIT license.

import multiprocessing as mp
import numpy as np
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
