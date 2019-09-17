#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

""" FILES """

MAX_AGE = sys.maxsize


""" READINGS """

# this is the minimum number of readings for a user attribute in a month.
MINIMUM_READINGS = 2

""" ATTRIBUTES """


HEALT_GROUP = [3, 4, 5, 6, 1,7,8,9]
SLEEP_GROUP = [10,15,16,18,19]
ACTIVITY_GROUP = [21,22]


""" TRAINING """

USER_TEST_PERC = 0.10


MAX_AGE_GROUPS = [55, 60, MAX_AGE]


CLASS_NUMBER = 3

MINIMUM_ACCURACY = 0.8
