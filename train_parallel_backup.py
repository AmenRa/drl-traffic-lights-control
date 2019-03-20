#!/usr/bin/python

import sys
import gc
import time
import pickle
from generate_routefile import generate_routefile
from plot_stats import plot_stats
from simulators.simulator_gru import Simulator as Simulator_Gru
from simulators.simulator_gru import make_step
from agents.GRU_Swish_DQNAgent import DQNAgent as GRU_Swish_DQNAgent

import multiprocessing as mp


def create_simulator(state_size, agent, max_steps, seed, mode):
    # Generate routefile dynamically
    generate_routefile(max_steps, seed, mode)
    # Create Simulator_Gru
    sim = Simulator_Gru(label=mode, sumocfg='environments/' + mode + '/tlcs_config_train.sumocfg', state_size=STATE_SIZE, max_steps=max_steps, agent=GRU_Swish_DQNAgent)
    return sim


def stop_simulator(sim):
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = sim.stop()

    return cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput


def do_step(sim, step):
    sim.do_step(step)
    return sim


# main entry point
if __name__ == "__main__":
    EPISODES = 3
    MAX_STEPS = 3600
    GUI = False
    pool = mp.Pool(mp.cpu_count() - 1)

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
        batch_size=BATCH_SIZE,
        name=NAME
    )

    # Stats
    REWARD_STORE = []
    AVG_WAIT_STORE = []
    THROUGHPUT_STORE = []
    AVG_INTERSECTION_QUEUE_STORE = []

    for episode in range(len(REWARD_STORE), EPISODES):
        print('\n')
        print("----- Starting episode: ", episode)
        epsilon = EPSILON - (episode / EPISODES)
        GRU_Swish_DQNAgent.epsilon = epsilon
        start_time = time.time()

        # Create simulators
        # SIM_LOW = create_simulator(state_size=STATE_SIZE, agent=GRU_Swish_DQNAgent, max_steps=MAX_STEPS, seed=episode, mode='low')
        # SIM_HIGH = create_simulator(state_size=STATE_SIZE, agent=GRU_Swish_DQNAgent, max_steps=MAX_STEPS, seed=episode, mode='high')
        # SIM_NORTH_SOUTH = create_simulator(state_size=STATE_SIZE, agent=GRU_Swish_DQNAgent, max_steps=MAX_STEPS, seed=episode, mode='north-south')
        # SIM_EAST_WEST = create_simulator(state_size=STATE_SIZE, agent=GRU_Swish_DQNAgent, max_steps=MAX_STEPS, seed=episode, mode='east-west')
        # SIMS = [SIM_LOW, SIM_HIGH, SIM_NORTH_SOUTH, SIM_EAST_WEST]

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []

        def do_loop_low(state_size, agent, max_steps, episode, return_dict):
            sim = create_simulator(state_size, agent, max_steps, episode, mode='low')
            for step in range(0, MAX_STEPS):
                sim.do_step(step)
            return_dict['low'] = sim.stop()

        def do_loop_high(state_size, agent, max_steps, episode, return_dict):
            sim = create_simulator(state_size, agent, max_steps, episode, mode='high')
            for step in range(0, MAX_STEPS):
                sim.do_step(step)
            return_dict['high'] = sim.stop()

        def do_loop_north_south(state_size, agent, max_steps, episode, return_dict):
            sim = create_simulator(state_size, agent, max_steps, episode, mode='north-south')
            for step in range(0, MAX_STEPS):
                sim.do_step(step)
            return_dict['north-south'] = sim.stop()

        def do_loop_east_west(state_size, agent, max_steps, episode, return_dict):
            sim = create_simulator(state_size, agent, max_steps, episode, mode='east-west')
            for step in range(0, MAX_STEPS):
                sim.do_step(step)
            return_dict['east-west'] = sim.stop()

        p_low = mp.Process(target=do_loop_low, args=(STATE_SIZE, GRU_Swish_DQNAgent, MAX_STEPS, episode, return_dict))
        jobs.append(p_low)
        p_low.start()

        p_high = mp.Process(target=do_loop_high, args=(STATE_SIZE, GRU_Swish_DQNAgent, MAX_STEPS, episode, return_dict))
        jobs.append(p_high)
        p_high.start()

        p_north_south = mp.Process(target=do_loop_north_south, args=(STATE_SIZE, GRU_Swish_DQNAgent, MAX_STEPS, episode, return_dict))
        jobs.append(p_north_south)
        p_north_south.start()

        p_east_west = mp.Process(target=do_loop_east_west, args=(STATE_SIZE, GRU_Swish_DQNAgent, MAX_STEPS, episode, return_dict))
        jobs.append(p_east_west)
        p_east_west.start()

        for proc in jobs:
            proc.join()

        print(len(GRU_Swish_DQNAgent.memory))

        REWARD_STORE.append(return_dict['low'][0])
        AVG_WAIT_STORE.append(return_dict['low'][1])
        AVG_INTERSECTION_QUEUE_STORE.append(return_dict['low'][2])
        THROUGHPUT_STORE.append(return_dict['low'][3])

        REWARD_STORE.append(return_dict['high'][0])
        AVG_WAIT_STORE.append(return_dict['high'][1])
        AVG_INTERSECTION_QUEUE_STORE.append(return_dict['high'][2])
        THROUGHPUT_STORE.append(return_dict['high'][3])

        REWARD_STORE.append(return_dict['north-south'][0])
        AVG_WAIT_STORE.append(return_dict['north-south'][1])
        AVG_INTERSECTION_QUEUE_STORE.append(return_dict['north-south'][2])
        THROUGHPUT_STORE.append(return_dict['north-south'][3])

        REWARD_STORE.append(return_dict['east-west'][0])
        AVG_WAIT_STORE.append(return_dict['east-west'][1])
        AVG_INTERSECTION_QUEUE_STORE.append(return_dict['east-west'][2])
        THROUGHPUT_STORE.append(return_dict['east-west'][3])

        # for step in range(0, MAX_STEPS):
            # SIM_LOW.do_step(step)
            # SIM_HIGH.do_step(step)
            # SIM_NORTH_SOUTH.do_step(step)
            # SIM_EAST_WEST.do_step(step)

            # input_list = list(zip(SIMS, [step, step, step, step]))
            # SIMS = pool.map(make_step, input_list)
            # SIM_LOW = SIMS[0]
            # SIM_HIGH = SIMS[1]
            # SIM_NORTH_SOUTH = SIMS[2]
            # SIM_EAST_WEST = SIMS[3]

            # do_step([SIM_LOW, step])
            # do_step([SIM_HIGH, step])
            # do_step([SIM_NORTH_SOUTH, step])
            # do_step([SIM_EAST_WEST, step])

            # p_low = mp.Process(target=SIM_LOW.do_step, args=(step,))
            # p_high = mp.Process(target=SIM_HIGH.do_step, args=(step,))
            # p_north_south = mp.Process(target=SIM_NORTH_SOUTH.do_step, args=(step,))
            # p_east_west = mp.Process(target=SIM_EAST_WEST.do_step, args=(step,))
            #
            # p_low.start()
            # p_high.start()
            # p_north_south.start()
            # p_east_west.start()
            #
            # p_low.join()
            # p_high.join()
            # p_north_south.join()
            # p_east_west.join()

            # SIMS = [pool.apply(do_step, args=(sim, step)) for sim in SIMS]
            #
            # pool.join()
            #
            # print(SIMS)

            # Train agent
            # if len(GRU_Swish_DQNAgent.memory) >= BATCH_SIZE:
            #     GRU_Swish_DQNAgent.replay()

        # pool.close()

        # cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = stop_simulator(SIM_LOW)
        # REWARD_STORE.append(cumulative_reward)
        # AVG_WAIT_STORE.append(avg_waiting_time)
        # THROUGHPUT_STORE.append(throughput)
        # AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)
        #
        # cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = stop_simulator(SIM_HIGH)
        #
        # print(cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput)
        #
        # REWARD_STORE.append(cumulative_reward)
        # AVG_WAIT_STORE.append(avg_waiting_time)
        # THROUGHPUT_STORE.append(throughput)
        # AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)
        #
        # cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = stop_simulator(SIM_NORTH_SOUTH)
        # REWARD_STORE.append(cumulative_reward)
        # AVG_WAIT_STORE.append(avg_waiting_time)
        # THROUGHPUT_STORE.append(throughput)
        # AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)
        #
        # cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = stop_simulator(SIM_EAST_WEST)
        # REWARD_STORE.append(cumulative_reward)
        # AVG_WAIT_STORE.append(avg_waiting_time)
        # THROUGHPUT_STORE.append(throughput)
        # AVG_INTERSECTION_QUEUE_STORE.append(avg_intersection_queue)

        print(REWARD_STORE, AVG_WAIT_STORE, THROUGHPUT_STORE, AVG_INTERSECTION_QUEUE_STORE)

        # del SIMS
        # del SIM_LOW
        # del SIM_HIGH
        # del SIM_NORTH_SOUTH
        # del SIM_EAST_WEST

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
