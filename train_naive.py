#!/anaconda3/bin/python

import sys
import gc
import time
import pickle
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulators.simulator_naive import Simulator as Simulator_Naive

# main entry point
if __name__ == "__main__":
    EPISODES = 2000
    MAX_STEPS = 3600
    SUMOCFG = "environments/high/tlcs_config_test.sumocfg"
    GUI = True

    # Stats
    REWARD_STORE = []
    AVG_WAIT_STORE = []
    THROUGHPUT_STORE = []
    AVG_INTERSECTION_QUEUE_STORE = []

    for episode in range(len(REWARD_STORE), EPISODES):
        print('\n')
        print("----- Starting episode: ", episode)
        start_time = time.time()
        # Generate routefile dynamically
        generate_routefile(max_steps=MAX_STEPS, seed=episode, mode='high')
        # Create Simulator_Naive
        SIM = Simulator_Naive(sumocfg=SUMOCFG)
        # Run simulator
        avg_waiting_time, avg_intersection_queue, throughput = SIM.run(gui=GUI, max_steps=MAX_STEPS)
        del SIM

        AVG_WAIT_STORE.append(avg_waiting_time)
        THROUGHPUT_STORE.append(throughput)
        AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)

        elapsed_time = round(time.time() - start_time, 2)
        print("----- Elapsed time: ", elapsed_time, " seconds -----")

        if (episode + 1) % 50 == 0:
            with open('history/gru_swish_AVG_WAIT_STORE.out', 'wb') as f:
                pickle.dump(AVG_WAIT_STORE, f)
            with open('history/gru_swish_THROUGHPUT_STORE.out', 'wb') as f:
                pickle.dump(THROUGHPUT_STORE, f)
            with open('history/gru_swish_AVG_INTERSECTION_QUEUE_STORE.out', 'wb') as f:
                pickle.dump(AVG_INTERSECTION_QUEUE_STORE, f)
            plot_stats('Naive', REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

    del GRU_Swish_DQNAgent
    gc.collect()

    sys.stdout.flush()
