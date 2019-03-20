#!/usr/bin/python3

import sys
import gc
from generate_routefile import generate_routefile
from plot_stats import plot_stats
# Simulators
from simulators.simulator_naive import Simulator as Simulator_Naive
from simulators.simulator import Simulator as Simulator
from simulators.simulator_gru import Simulator as Simulator_Gru
# Agents
from agents.ReLU_DQNAgent import DQNAgent as ReLU_DQNAgent
from agents.Swish_DQNAgent import DQNAgent as Swish_DQNAgent
from agents.GRU_ReLU_DQNAgent import DQNAgent as GRU_ReLU_DQNAgent
from agents.GRU_Swish_DQNAgent import DQNAgent as GRU_Swish_DQNAgent


def test_naive(MAX_STEPS, SUMOCFG, GUI):
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


def test_ff(MAX_STEPS, SUMOCFG, GUI, AGENT):
    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator
    SIM = Simulator(sumocfg=SUMOCFG, state_size=STATE_SIZE, agent=AGENT)
    # Run simulator
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = SIM.run(gui=GUI, max_steps=MAX_STEPS)
    del SIM
    print(AGENT.name)
    print(avg_waiting_time)
    print(avg_intersection_queue)
    print(throughput)


def test_gru(MAX_STEPS, SUMOCFG, GUI, AGENT):
    avg_waiting_time = 0
    avg_intersection_queue = 0
    throughput = 0
    # Create Simulator
    SIM = Simulator_Gru(sumocfg=SUMOCFG, state_size=STATE_SIZE, agent=AGENT)
    # Run simulator
    cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput = SIM.run(gui=GUI, max_steps=MAX_STEPS)
    del SIM
    print(AGENT.name)
    print(avg_waiting_time)
    print(avg_intersection_queue)
    print(throughput)


def test(SEED, MAX_STEPS, SUMOCFG, GUI, ReLU_DQNAgent, Swish_DQNAgent, GRU_ReLU_DQNAgent, GRU_Swish_DQNAgent):
    # Generate routefile dynamically
    generate_routefile(max_steps=MAX_STEPS, seed=1)
    test_naive(MAX_STEPS, SUMOCFG, GUI)
    test_ff(MAX_STEPS, SUMOCFG, GUI, AGENT=ReLU_DQNAgent)
    # test_ff(MAX_STEPS, SUMOCFG, GUI, AGENT=Swish_DQNAgent)
    # test_gru(MAX_STEPS, SUMOCFG, GUI, AGENT=GRU_ReLU_DQNAgent)
    # test_gru(MAX_STEPS, SUMOCFG, GUI, AGENT=GRU_Swish_DQNAgent)


# main entry point
if __name__ == "__main__":
    MAX_STEPS = 3600
    SUMOCFG = "environment/tlcs_config_train.sumocfg"
    GUI = True

    # Agent hyperparameters
    STATE_SIZE = 321
    ACTION_SIZE = 4
    MEMORY_SIZE = 200
    GAMMA = 0.95
    EPSILON = 0.
    EPSILON_DECAY_RATE = 0.99999
    EPSILON_MIN = 0.01
    LEARNING_RATE = 0.0002

    # Create ReLU_DQNAgent
    ReLU_DQNAgent = ReLU_DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        memory_size=MEMORY_SIZE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        epsilon_min=EPSILON_MIN,
        learning_rate=LEARNING_RATE,
        name='ReLU_DQNAgent'
    )
    # ReLU_DQNAgent = ReLU_DQNAgent.load()

    # Create Swish_DQNAgent
    Swish_DQNAgent = Swish_DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        memory_size=MEMORY_SIZE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        epsilon_min=EPSILON_MIN,
        learning_rate=LEARNING_RATE,
        name='Swish_DQNAgent'
    )
    # Swish_DQNAgent = Swish_DQNAgent.load()

    # Create GRU_ReLU_DQNAgent
    GRU_ReLU_DQNAgent = GRU_ReLU_DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        memory_size=MEMORY_SIZE,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        epsilon_min=EPSILON_MIN,
        learning_rate=LEARNING_RATE,
        name='GRU_ReLU_DQNAgent'
    )
    # GRU_ReLU_DQNAgent = GRU_ReLU_DQNAgent.load()

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
        name='GRU_Swish_DQNAgent'
    )
    # GRU_Swish_DQNAgent = GRU_Swish_DQNAgent.load()

    # Low traffic test
    test(0, MAX_STEPS, SUMOCFG, GUI, ReLU_DQNAgent, Swish_DQNAgent, GRU_ReLU_DQNAgent, GRU_Swish_DQNAgent)

    # High traffic test
    test(1, MAX_STEPS, SUMOCFG, GUI, ReLU_DQNAgent, Swish_DQNAgent, GRU_ReLU_DQNAgent, GRU_Swish_DQNAgent)

    # Nort-South Main traffic test
    test(2, MAX_STEPS, SUMOCFG, GUI, ReLU_DQNAgent, Swish_DQNAgent, GRU_ReLU_DQNAgent, GRU_Swish_DQNAgent)

    # East-West Main traffic test
    test(3, MAX_STEPS, SUMOCFG, GUI, ReLU_DQNAgent, Swish_DQNAgent, GRU_ReLU_DQNAgent, GRU_Swish_DQNAgent)

    del ReLU_DQNAgent
    del Swish_DQNAgent
    del GRU_ReLU_DQNAgent
    del GRU_Swish_DQNAgent

    gc.collect()

    sys.stdout.flush()
