#!/usr/bin/python3

import sys
import gc
import time
import pickle
import requests
from ast import literal_eval
from flask import Flask, request, jsonify
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulators.simulator_gru import Simulator as Simulator_Gru
# from agents.GRU_Swish_DQNAgent import DQNAgent as GRU_Swish_DQNAgent

import multiprocessing as mp


def create_simulator(state_size, agent, max_steps, seed, mode):
    # Generate routefile dynamically
    generate_routefile(max_steps, seed, mode)
    # Create Simulator_Gru
    sim = Simulator_Gru(label=mode, sumocfg='environments/' + mode + '/tlcs_config_train.sumocfg', state_size=STATE_SIZE, max_steps=max_steps, agent=agent)
    return sim


def simulate(mode, state_size, agent, max_steps, episode, return_dict):
    sim = create_simulator(state_size, agent, max_steps, episode, mode)
    for step in range(0, max_steps):
        sim.do_step(step)
    return_dict[mode] = sim.stop()


# main entry point
if __name__ == "__main__":
    EPISODES = 100
    MAX_STEPS = 3600
    # pool = mp.Pool(mp.cpu_count() - 1)

    # Agent hyperparameters
    STATE_SIZE = 321
    ACTION_SIZE = 4
    MEMORY_SIZE = 500
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_DECAY_RATE = 0.99999
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.0002
    BATCH_SIZE = 32
    NAME = 'GRU_Swish_DQNAgent'

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
            p = mp.Process(target=simulate, args=(mode, STATE_SIZE, 'ciao', MAX_STEPS, episode, return_dict))
            jobs.append(p)
            p.start()

        # p_low = mp.Process(target=simulate, args=('low', STATE_SIZE, 'ciao', MAX_STEPS, episode, return_dict))
        # jobs.append(p_low)
        # p_low.start()

        # p_high = mp.Process(target=simulate, args=('high', STATE_SIZE, 'ciao', MAX_STEPS, episode, return_dict))
        # jobs.append(p_high)
        # p_high.start()
        #
        # p_north_south = mp.Process(target=simulate, args=('north-south', STATE_SIZE, 'ciao', MAX_STEPS, episode, return_dict))
        # jobs.append(p_north_south)
        # p_north_south.start()
        #
        # p_east_west = mp.Process(target=simulate, args=('east-west', STATE_SIZE, 'ciao', MAX_STEPS, episode, return_dict))
        # jobs.append(p_east_west)
        # p_east_west.start()

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

        if (episode + 1) % 50 == 0:
            requests.post('http://127.0.0.1:5000/save')
            with open('history/gru_swish_REWARD_STORE.out', 'wb') as f:
                pickle.dump(REWARD_STORE, f)
            with open('history/gru_swish_AVG_WAIT_STORE.out', 'wb') as f:
                pickle.dump(AVG_WAIT_STORE, f)
            with open('history/gru_swish_THROUGHPUT_STORE.out', 'wb') as f:
                pickle.dump(THROUGHPUT_STORE, f)
            with open('history/gru_swish_AVG_INTERSECTION_QUEUE_STORE.out', 'wb') as f:
                pickle.dump(AVG_INTERSECTION_QUEUE_STORE, f)
            plot_stats(NAME, REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

    # del GRU_Swish_DQNAgent
    # gc.collect()

    sys.stdout.flush()