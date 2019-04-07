#!/usr/bin/python3

import sys
import gc
from generate_routefile import generate_routefile
# Simulators
from simulators.simulator_naive_2 import Simulator as Simulator_Naive
from results.feed_forward_dropout.simulator_test import Simulator as Simulator_FFNO


def test_naive(sumocfg, gui, state_size, max_steps):
    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator_Naive
    sim = Simulator_Naive('naive', sumocfg, max_steps, gui)
    # Run simulator
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = sim.run(max_steps)
    del sim
    return avg_waiting_time, avg_intersection_queue, throughput


def test_ffno(sumocfg, gui, state_size, max_steps):
    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator_FFNO
    sim = Simulator_FFNO('ffno', sumocfg, state_size, max_steps, gui)
    # Run simulator
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = sim.run(max_steps)
    del sim
    return avg_waiting_time, avg_intersection_queue, throughput


# main entry point
if __name__ == "__main__":
    SUMOCFG = "environments/test/tlcs_config_test.sumocfg"
    MAX_STEPS = 3600
    GUI = False
    STATE_SIZE = 320

    w = []
    q = []
    t = []

    for mode in ['low', 'high', 'north-south', 'east-west']:
        sumocfg = 'environments/' + mode + '/tlcs_config_train.sumocfg'
        generate_routefile(MAX_STEPS, 666, mode)

        # print('\nMethod: Naive')
        # avg_waiting_time, avg_intersection_queue, throughput = test_naive(sumocfg, GUI, STATE_SIZE, MAX_STEPS)
        # w.append(round(avg_waiting_time, 2))
        # q.append(round(avg_intersection_queue, 2))
        # t.append(round(throughput, 2))

        print('\nMethod: Feed-Forward Dropout')
        avg_waiting_time, avg_intersection_queue, throughput = test_ffno(sumocfg, GUI, STATE_SIZE, MAX_STEPS)
        w.append(round(avg_waiting_time, 2))
        q.append(round(avg_intersection_queue, 2))
        t.append(round(throughput, 2))

        print('\n')

    for a, b, c in zip(w, q, t):
        print(a, b, c)

    gc.collect()

    sys.stdout.flush()
