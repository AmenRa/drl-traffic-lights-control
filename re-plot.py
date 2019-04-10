import pickle
from plot_stats import plot_stats

# Stats
REWARD_STORE = []
AVG_WAIT_STORE = []
THROUGHPUT_STORE = []
AVG_INTERSECTION_QUEUE_STORE = []

with open('history/REWARD_STORE.out', 'rb') as f:
    REWARD_STORE = pickle.load(f)
with open('history/AVG_WAIT_STORE.out', 'rb') as f:
    AVG_WAIT_STORE = pickle.load(f)
with open('history/THROUGHPUT_STORE.out', 'rb') as f:
    THROUGHPUT_STORE = pickle.load(f)
with open('history/AVG_INTERSECTION_QUEUE_STORE.out', 'rb') as f:
    AVG_INTERSECTION_QUEUE_STORE = pickle.load(f)

plot_stats('Feed-Forward Dropout DQNAgent', REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)
