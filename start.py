#!/usr/bin/python3

from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulator import Simulator
from agent import DQNAgent
import os
import sys
import gc
import time

# to kill warning about tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# main entry point
if __name__ == "__main__":
    EPISODES = 100
    MAX_STEPS = 5400
    # sumocfg = "environment/model.sumocfg"
    SUMOCFG = "environment/tlcs_config_test.sumocfg"
    # sumocfg = "environment/tlcs_config_train.sumocfg"
    TRIPINFO = "environment/tripinfo.xml"
    GUI = False

    # Agent hyperparameters
    STATE_SIZE = 200
    ACTION_SIZE = 4
    MEMORY_SIZE = 200
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_DECAY_RATE = 0.9999
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.0002
    NAME = 'DQNAgent'

    # Stats
    reward_store = []
    avg_wait_store = []
    throughput_store = []
    avg_intersection_queue_store = []

    # Create DQNAgent
    AGENT = DQNAgent(
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

    for episode in range(EPISODES):
        print('********************')
        print("----- Starting episode: ", episode)
        start_time = time.time()
        # Generate routefile dynamically
        generate_routefile(max_steps=MAX_STEPS, seed=42)
        # Create Simulator
        SIM = Simulator(sumocfg=SUMOCFG, tripinfo=TRIPINFO, state_size=STATE_SIZE, agent=AGENT)
        # Run simulator
        cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = SIM.run(gui=GUI, max_steps=MAX_STEPS)
        del SIM

        reward_store.append(cumulative_reward)
        avg_wait_store.append(avg_waiting_time)
        throughput_store.append(throughput)
        avg_intersection_queue_store.append(avg_intersection_queue)

        elapsed_time = round(time.time() - start_time, 2)
        print("----- Elapsed time: ", elapsed_time, " seconds -----")
        print('********************')

    del AGENT
    gc.collect()

    plot_stats(reward_store, avg_wait_store, throughput_store, avg_intersection_queue_store)

    sys.stdout.flush()
