import os
import sys
# This module exports a set of functions implemented in C corresponding to the intrinsic operators of Python. For example, operator.add(x, y) is equivalent to the expression x+y. The function names are those used for special methods; variants without leading and trailing '__' are also provided for convenience.
import operator
from functools import reduce
import numpy as np

import time

# Import some Python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    TOOLS = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(TOOLS)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Disable come pylint options because the following packages are loaded from SUMO_HOME
# pylint: disable=E0401,C0413
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import traci.constants as tc
# pylint: enable=E0401,C0413

# Duration of green phase
GREEN_PHASE_DURATION = 31
# Duration of yellow phase
YELLOW_PHASE_DURATION = 6

# phase codes based on xai_tlcs.net.xml
PHASE_NS_GREEN = 0 # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2 # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4 # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6 # action 3 code 11
PHASE_EWL_YELLOW = 7

def compute_queue_and_total_waiting_time(vehicle):
    in_queue = False
    waiting_time = vehicle[1][122]
    # Lane position
    lane_pos = vehicle[1][86]
    # Lane ID
    lane_id = vehicle[1][81]
    # Lanes are 750m long, this calculate the distance from the tls
    distance_from_tls = 750 - lane_pos
    # Lane group initialization
    lane_group = -1
    # Flag to leave out vehicles crossing the intersection or driving away from it
    valid_car = False

    # In which lane is the car? _3 are the "turn left only" lanes
    if lane_id in ('W2TL_0', 'W2TL_1', 'W2TL_2'):
        lane_group = 0
    elif lane_id == 'W2TL_3':
        lane_group = 1
    elif lane_id in ('N2TL_0', 'N2TL_1', 'N2TL_2'):
        lane_group = 2
    elif lane_id == 'N2TL_3':
        lane_group = 3
    elif lane_id in ('E2TL_0', 'E2TL_1', 'E2TL_2'):
        lane_group = 4
    elif lane_id == 'E2TL_3':
        lane_group = 5
    elif lane_id in ('S2TL_0', 'S2TL_1', 'S2TL_2'):
        lane_group = 6
    elif lane_id == 'S2TL_3':
        lane_group = 7

    # distance in meters from the TLS -> mapping into cells
    if distance_from_tls < 7:
        lane_cell = 0
    elif distance_from_tls < 14:
        lane_cell = 1
    elif distance_from_tls < 21:
        lane_cell = 2
    elif distance_from_tls < 28:
        lane_cell = 3
    elif distance_from_tls < 40:
        lane_cell = 4
    elif distance_from_tls < 60:
        lane_cell = 5
    elif distance_from_tls < 100:
        lane_cell = 6
    elif distance_from_tls < 160:
        lane_cell = 7
    elif distance_from_tls < 400:
        lane_cell = 8
    elif distance_from_tls <= 750:
        lane_cell = 9

    if 1 <= lane_group <= 7:
        # composition of the two postion ID to create a number in interval 0-79
        vehicle_position = int(str(lane_group) + str(lane_cell))
        valid_car = True
    elif lane_group == 0:
        vehicle_position = lane_cell
        valid_car = True
    else:
        vehicle_position = -1

    if valid_car:
        # Heuristic to capture with precision when a car is really in queue
        if vehicle[1][122] > 0.5:
            in_queue = True

    return in_queue, waiting_time, vehicle_position, valid_car


class Simulator:
    def __init__(self, sumocfg, tripinfo, state_size, agent):
        self.sumocfg = sumocfg
        self.tripinfo = tripinfo
        self.state_size = state_size
        self.agent = agent

    def _compute_throughput(self):
        return traci.simulation.getArrivedNumber()

    def sum_vectors_elementwise(self, v1, v2):
        return v1 + v2

    def get_state(self, junction_id):
        subscription_results = traci.junction.getContextSubscriptionResults(junction_id)

        state = np.zeros(self.state_size)
        current_cumulated_waiting_time = 0
        current_queue = 0
        current_throughput = 0

        if subscription_results is not None:
            vehicles = subscription_results.items()

            in_queues, waiting_times, vehicle_positions, valid_cars = zip(*map(compute_queue_and_total_waiting_time, vehicles))

            current_cumulated_waiting_time = reduce(operator.add, waiting_times)
            current_queue = reduce(operator.add, in_queues)
            # current_throughput = self._compute_throughput(vehicles)
            for vehicle_position in vehicle_positions:
                if vehicle_position > -1:
                    state[vehicle_position] = 1

        return state, current_cumulated_waiting_time, current_queue, current_throughput

    # Calculate reward
    def get_reward(self, step, current_waiting_time):
        if current_waiting_time > 0:
            return 1 / current_waiting_time
        # Avoid to return 1 in the first steps when no vehicles are near the tls
        if step < 50:
            return 0
        return 1

    # Check if the episode is finished (not very useful here, but often used in Reinforcement Learning tasks)
    def is_done(self, step, max_steps):
        return step < max_steps - 1

    # Run simulation (TraCI/SUMO)
    def run(self, gui=False, max_steps=100, batch_size=32):
        # Control SUMO mode (with or without GUI)
        if gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        # Start SUMO with TraCI and some flags
        traci.start([sumo_binary, "-c", self.sumocfg, "--no-step-log", "true", "--tripinfo-output", self.tripinfo])

        junction_id = "TL"

        # The following code retrieves all vehicle speeds and waiting times within range (50m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
        # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
        traci.junction.subscribeContext(junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000, [tc.VAR_LANEPOSITION, tc.VAR_LANE_ID, tc.VAR_WAITING_TIME])
        # tc.VAR_SPEED, tc.VAR_WAITING_TIME,
        # 86 : VAR_LANEPOSITION
        # 81 : E2TL_0
        # 122 : VAR_WAITING_TIME
        # 121 : VAR_ARRIVED_VEHICLES_NUMBER

        # stats
        cumulative_reward = 0
        cumulative_waiting_time = 0
        throughput = 0
        cumulative_intersection_queue = 0

        state, previous_waiting_time, intersection_queue, current_throughput = self.get_state(junction_id)

        yellow_phase = False
        green_phase = False
        yellow_phase_step_count = 0
        green_phase_step_count = 0

        action = None
        previous_action = None

        for step in range(max_steps):
            # Choose action and (start yellow phase or execute action)
            if not green_phase and not yellow_phase:
                # Let the agent choose action
                # Store previous action
                previous_action = action
                # Choose action
                action = self.agent.act(state)
                # Start yellow phase
                if action != previous_action and previous_action is not None:
                    yellow_phase = True
                    # print('start yellow phase')
                    # yellow phase number based on previous action
                    traci.trafficlight.setPhase("TL", previous_action * 2 + 1)
                # Execute action
                else:
                    # print('start green phase')
                    green_phase = True
                    if action == 0:
                        traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
                    elif action == 1:
                        traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
                    elif action == 2:
                        traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
                    elif action == 3:
                        traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

            # Do step
            traci.simulationStep(step)

            # Update yellow phase counter
            if yellow_phase:
                # print('update yellow phase counter')
                yellow_phase_step_count += 1
                # Reset yellow phase
                if yellow_phase_step_count == YELLOW_PHASE_DURATION:
                    # print('reset yellow phase count')
                    yellow_phase = False
                    yellow_phase_step_count = 0
            # Update green phase counter
            elif green_phase:
                # print('update green phase counter')
                green_phase_step_count += 1
                # Reset green phase
                if green_phase_step_count == GREEN_PHASE_DURATION:
                    # print('reset green phase count')
                    green_phase = False
                    green_phase_step_count = 0
                elif green_phase_step_count == 1:
                    # print('do things')
                    next_state, current_waiting_time, intersection_queue, current_throughput = self.get_state(junction_id)
                    reward = self.get_reward(step, current_waiting_time)
                    done = self.is_done(step, max_steps)
                    # Feed agent memory
                    self.agent.remember(state, action, reward, next_state, done)

                    # Update
                    state = next_state
                    cumulative_reward += reward
                    cumulative_waiting_time += current_waiting_time
                    cumulative_intersection_queue += intersection_queue

                    # Train agent
                    if len(self.agent.memory) >= 32:
                        self.agent.replay(batch_size)

            throughput += self._compute_throughput()

        print("Total reward: {}, Eps: {}".format(cumulative_reward, self.agent.epsilon))

        traci.close(False)

        # Return the stats for this episode
        avg_waiting_time = cumulative_waiting_time / max_steps
        avg_intersection_queue = cumulative_intersection_queue / max_steps

        return cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput
