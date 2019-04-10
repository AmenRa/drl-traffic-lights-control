#!/usr/bin/python3

import sys
import time
import pickle
import requests
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from lstm_dropout.simulator_train import Simulator

import multiprocessing as mp


def create_simulator(state_size, max_steps, seed, mode):
    # Generate routefile dynamically
    generate_routefile(max_steps, seed, mode)
    # Create Simulator
    sim = Simulator(label=mode, sumocfg='environments/' + mode + '/tlcs_config_train.sumocfg', state_size=STATE_SIZE, max_steps=max_steps)
    return sim


def simulate(mode, state_size, max_steps, episode, return_dict):
    sim = create_simulator(state_size, max_steps, episode, mode)
    for step in range(0, max_steps):
        sim.do_step(step)
    return_dict[mode] = sim.stop()


# main entry point
if __name__ == "__main__":
    EPISODES = 100
    MAX_STEPS = 3600
    # pool = mp.Pool(mp.cpu_count() - 1)

    # Agent hyperparameters
    STATE_SIZE = 320
    EPSILON = 1.0
    NAME = 'LSTM Dropout DQNAgent'

    # Stats
    REWARD_STORE = []
    AVG_WAIT_STORE = []
    THROUGHPUT_STORE = []
    AVG_INTERSECTION_QUEUE_STORE = []

    for episode in range(len(REWARD_STORE), EPISODES):
        print('\n')
        print("----- Starting episode: ", episode)

        epsilon = EPSILON - (episode / EPISODES)
        requests.post('http://127.0.0.1:5000/update_epsilon', json={'epsilon': epsilon})

        start_time = time.time()

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []

        # for mode in ['low']:
        for mode in ['low', 'high', 'north-south', 'east-west']:
            p = mp.Process(target=simulate, args=(mode, STATE_SIZE, MAX_STEPS, episode, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        for key in ['low', 'high', 'north-south', 'east-west']:
            REWARD_STORE.append(return_dict[key][0])
            AVG_WAIT_STORE.append(return_dict[key][1])
            AVG_INTERSECTION_QUEUE_STORE.append(return_dict[key][2])
            THROUGHPUT_STORE.append(return_dict[key][3])

        # Experience
        requests.post('http://127.0.0.1:5000/replay')

        elapsed_time = round(time.time() - start_time, 2)
        print("----- Elapsed time: ", elapsed_time, " seconds -----")

        if (episode + 1) % 10 == 0:
            requests.post('http://127.0.0.1:5000/save')
            with open('history/REWARD_STORE.out', 'wb') as f:
                pickle.dump(REWARD_STORE, f)
            with open('history/AVG_WAIT_STORE.out', 'wb') as f:
                pickle.dump(AVG_WAIT_STORE, f)
            with open('history/THROUGHPUT_STORE.out', 'wb') as f:
                pickle.dump(THROUGHPUT_STORE, f)
            with open('history/AVG_INTERSECTION_QUEUE_STORE.out', 'wb') as f:
                pickle.dump(AVG_INTERSECTION_QUEUE_STORE, f)
            plot_stats(NAME, REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

    sys.stdout.flush()
