#!/usr/bin/python3

import pickle
from plot_stats import plot_stats

NAME = 'Feed-Forward-Dropout'

# Stats
REWARD_STORE = []
AVG_WAIT_STORE = []
THROUGHPUT_STORE = []
AVG_INTERSECTION_QUEUE_STORE = []

with open('history/ffdo_REWARD_STORE.out', 'rb') as f:
    REWARD_STORE = pickle.load(f)
with open('history/ffdo_AVG_WAIT_STORE.out', 'rb') as f:
    AVG_WAIT_STORE = pickle.load(f)
with open('history/ffdo_THROUGHPUT_STORE.out', 'rb') as f:
    THROUGHPUT_STORE = pickle.load(f)
with open('history/ffdo_AVG_INTERSECTION_QUEUE_STORE.out', 'rb') as f:
    AVG_INTERSECTION_QUEUE_STORE = pickle.load(f)

# print(REWARD_STORE)
# print(AVG_WAIT_STORE)
# print(THROUGHPUT_STORE)
# print(AVG_INTERSECTION_QUEUE_STORE)

plot_stats('NON_LO_SO', REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

# plot_stats(NAME, REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)