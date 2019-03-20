#!/usr/bin/python3

import sys
import gc
from generate_routefile import generate_routefile
from plot_stats import plot_stats
# Simulators
from simulators.simulator_naive import Simulator as Simulator_Naive

if __name__ == "__main__":
    MAX_STEPS = 3600
    SUMOCFG = "environment/tlcs_config_train.sumocfg"
    GUI = True

    generate_routefile(max_steps=MAX_STEPS, seed=1)

    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator
    SIM_NAIVE = Simulator_Naive(sumocfg=SUMOCFG)
    # Run simulator
    avg_waiting_time, avg_intersection_queue, throughput = SIM_NAIVE.run(gui=GUI, max_steps=MAX_STEPS)
    del SIM_NAIVE
    print('Naive')
    print(avg_waiting_time)
    print(avg_intersection_queue)
    print(throughput)
