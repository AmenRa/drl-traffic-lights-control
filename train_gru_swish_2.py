#!/usr/bin/python3

import sys
import gc
import time
import pickle
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulators.simulator_gru import Simulator as Simulator_Gru
from simulators.simulator_gru import extract_waiting_time, is_queued, compute_position_index, compute_car_state
from agents.GRU_Swish_DQNAgent import DQNAgent as GRU_Swish_DQNAgent

# main entry point
if __name__ == "__main__":
    EPISODES = 100
    MAX_STEPS = 3600
    SUMOCFG = "environment/tlcs_config_train.sumocfg"
    GUI = False

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

    with open('history/gru_swish_REWARD_STORE.out', 'rb') as f:
        REWARD_STORE = pickle.load(f)
    with open('history/gru_swish_AVG_WAIT_STORE.out', 'rb') as f:
        AVG_WAIT_STORE = pickle.load(f)
    with open('history/gru_swish_THROUGHPUT_STORE.out', 'rb') as f:
        THROUGHPUT_STORE = pickle.load(f)
    with open('history/gru_swish_AVG_INTERSECTION_QUEUE_STORE.out', 'rb') as f:
        AVG_INTERSECTION_QUEUE_STORE = pickle.load(f)

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

    # GRU_Swish_DQNAgent.act([])

    GRU_Swish_DQNAgent.load()

    for episode in range(len(REWARD_STORE), EPISODES):
        print('\n')
        print("----- Starting episode: ", episode)
        GRU_Swish_DQNAgent.epsilon = EPSILON - (episode / EPISODES)
        start_time = time.time()
        # Generate routefile dynamically
        generate_routefile(max_steps=MAX_STEPS, seed=episode)
        # Create Simulator_Gru
        SIM = Simulator_Gru(sumocfg=SUMOCFG, state_size=STATE_SIZE, agent=GRU_Swish_DQNAgent)
        # Run simulator
        cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = SIM.run(gui=GUI, max_steps=MAX_STEPS)
        del SIM

        REWARD_STORE.append(cumulative_reward)
        AVG_WAIT_STORE.append(avg_waiting_time)
        THROUGHPUT_STORE.append(throughput)
        AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)

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
