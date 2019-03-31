#!/usr/bin/python3

import sys
import gc
from generate_routefile import generate_routefile
from plot_stats import plot_stats
# Simulators
from simulators.simulator_naive_2 import Simulator as Simulator_Naive
from results.feed_forward_dropout.simulator import Simulator as Simulator_FFNO


def test_naive(sumocfg, gui, state_size, max_steps):
    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator_Naive
    sim = Simulator_Naive('naive', sumocfg, max_steps, gui)
    # Run simulator
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = sim.run(max_steps)
    del sim
    print('Naive')
    print('Waiting time:', avg_waiting_time)
    print('Queue:', avg_intersection_queue)
    print('Throughput:', throughput)


def test_ffno(sumocfg, gui, state_size, max_steps):
    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator_FFNO
    sim = Simulator_FFNO('ffno', sumocfg, state_size, max_steps, gui)
    # Run simulator
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = sim.run(max_steps)
    del sim
    print('Feed-Forward Dropout Network')
    print('Waiting time:', avg_waiting_time)
    print('Queue:', avg_intersection_queue)
    print('Throughput:', throughput)


# main entry point
if __name__ == "__main__":
    SUMOCFG = "environments/test/tlcs_config_test.sumocfg"
    MAX_STEPS = 5400
    GUI = False
    STATE_SIZE = 320

    # test_naive(SUMOCFG, GUI, STATE_SIZE, MAX_STEPS)
    test_ffno(SUMOCFG, GUI, STATE_SIZE, MAX_STEPS)

    gc.collect()

    sys.stdout.flush()
