#!/anaconda3/bin/python

import os
import sys
import gc
import time
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulator import Simulator
from agent import DQNAgent

# to kill warning about tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# main entry point
if __name__ == "__main__":
    EPISODES = 1
    MAX_STEPS = 3600
    # sumocfg = "environment/model.sumocfg"
    SUMOCFG = "environment/tlcs_config_test.sumocfg"
    # sumocfg = "environment/tlcs_config_train.sumocfg"
    TRIPINFO = "environment/tripinfo.xml"
    GUI = False

    # Agent hyperparameters
    STATE_SIZE = 80
    ACTION_SIZE = 4
    MEMORY_SIZE = 200
    GAMMA = 0.95
    EPSILON = 1.0
    EPSILON_DECAY_RATE = 0.99999
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.0002
    NAME = 'DQNAgent'

    # Stats
    REWARD_STORE = []
    AVG_WAIT_STORE = []
    THROUGHPUT_STORE = []
    AVG_INTERSECTION_QUEUE_STORE = []

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

        REWARD_STORE.append(cumulative_reward)
        AVG_WAIT_STORE.append(avg_waiting_time)
        THROUGHPUT_STORE.append(throughput)
        AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)

        elapsed_time = round(time.time() - start_time, 2)
        print("----- Elapsed time: ", elapsed_time, " seconds -----")
        print('********************')

    AGENT.save()
    del AGENT
    gc.collect()

    plot_stats(REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

    sys.stdout.flush()
