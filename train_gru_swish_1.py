#!/usr/bin/python

import sys
import gc
import time
import pickle
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulators.simulator_gru import Simulator as Simulator_Gru
from agents.GRU_Swish_DQNAgent import DQNAgent as GRU_Swish_DQNAgent

# main entry point
if __name__ == "__main__":
    EPISODES = 100
    MAX_STEPS = 3600
    SUMOCFG = "environment/tlcs_config_train.sumocfg"
    GUI = False
    BATCH_SIZE = 32

    # Agent hyperparameters
    STATE_SIZE = 321
    ACTION_SIZE = 4
    MEMORY_SIZE = 200
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_DECAY_RATE = 0.99999
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.0002
    NAME = 'GRU_Swish_DQNAgent'

    # Stats
    REWARD_STORE = []
    AVG_WAIT_STORE = []
    THROUGHPUT_STORE = []
    AVG_INTERSECTION_QUEUE_STORE = []

    # Create GRU_Swish_DQNAgent
    GRU_Swish_DQNAgent = GRU_Swish_DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        memory_size=MEMORY_SIZE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        epsilon_min=EPSILON_MIN,
        learning_rate=LEARNING_RATE,
        name=NAME
    )

    for episode in range(len(REWARD_STORE), EPISODES):
        print('\n')
        print("----- Starting episode: ", episode)
        epsilon = EPSILON - (episode / EPISODES)
        GRU_Swish_DQNAgent.epsilon = epsilon
        start_time = time.time()
        # Generate routefile dynamically
        generate_routefile(max_steps=MAX_STEPS, seed=5)
        # Create Simulator_Gru
        SIM = Simulator_Gru(sumocfg=SUMOCFG, state_size=STATE_SIZE, agent=GRU_Swish_DQNAgent)
        # Run simulator
        cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = SIM.run(gui=GUI, max_steps=MAX_STEPS, batch_size=BATCH_SIZE)
        del SIM

        REWARD_STORE.append(cumulative_reward)
        AVG_WAIT_STORE.append(avg_waiting_time)
        THROUGHPUT_STORE.append(throughput)
        AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)

        print("Cumulative reward: {}\nAvarage waiting time: {}\nThroughput: {}\nAvarage intersection queue: {}\nEpsilon: {}".format(cumulative_reward, avg_waiting_time, throughput, avg_intersection_queue, epsilon))

        elapsed_time = round(time.time() - start_time, 2)
        print("----- Elapsed time: ", elapsed_time, " seconds -----")

        if (episode + 1) % 50 == 0:
            GRU_Swish_DQNAgent.save()
            with open('history/gru_swish_REWARD_STORE.out', 'wb') as f:
                pickle.dump(REWARD_STORE, f)
            with open('history/gru_swish_AVG_WAIT_STORE.out', 'wb') as f:
                pickle.dump(AVG_WAIT_STORE, f)
            with open('history/gru_swish_THROUGHPUT_STORE.out', 'wb') as f:
                pickle.dump(THROUGHPUT_STORE, f)
            with open('history/gru_swish_AVG_INTERSECTION_QUEUE_STORE.out', 'wb') as f:
                pickle.dump(AVG_INTERSECTION_QUEUE_STORE, f)
            plot_stats(NAME, REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

    del GRU_Swish_DQNAgent
    gc.collect()

    sys.stdout.flush()
