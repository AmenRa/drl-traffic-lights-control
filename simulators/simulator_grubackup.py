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

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import traci.constants as tc

# Duration of green phase
GREEN_PHASE_DURATION = 21
# Duration of yellow phase
YELLOW_PHASE_DURATION = 6

# phase codes based on xai_tlcs.net.xml
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7


def make_step(sim, step):
    # Choose sim.action and (start yellow phase or execute sim.action)
    if not sim.green_phase and not sim.yellow_phase:
        # Let the agent choose action
        # Store previous action
        sim.previous_action = sim.action
        # Choose action
        sim.action = sim.agent.act(sim.states)
        # Start yellow phase
        if sim.action != sim.previous_action and sim.previous_action is not None:
            sim.yellow_phase = True
            # print('start yellow phase')
            # yellow phase number based on previous action
            sim.connection.trafficlight.setPhase("TL", sim.previous_action * 2 + 1)
        # Execute action
        else:
            # print('start green phase')
            sim.green_phase = True
            if sim.action == 0:
                sim.connection.trafficlight.setPhase("TL", PHASE_NS_GREEN)
            elif sim.action == 1:
                sim.connection.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
            elif sim.action == 2:
                sim.connection.trafficlight.setPhase("TL", PHASE_EW_GREEN)
            elif sim.action == 3:
                sim.connection.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    # Do step
    sim.connection.simulationStep(float(step))

    # Update yellow phase counter
    if sim.yellow_phase:
        # print('update yellow phase counter')
        sim.yellow_phase_step_count += 1
        # Reset yellow phase
        if sim.yellow_phase_step_count == YELLOW_PHASE_DURATION:
            # print('reset yellow phase count')
            sim.yellow_phase = False
            sim.yellow_phase_step_count = 0
        # Update green phase counter
        elif sim.green_phase:
            # print('update green phase counter')
            sim.green_phase_step_count += 1
        # Reset green phase
        if sim.green_phase_step_count == GREEN_PHASE_DURATION:
            # print('reset green phase count')
            sim.green_phase = False
            sim.green_phase_step_count = 0
        elif sim.green_phase_step_count == 1:
            # print('do things')
            current_waiting_time = sim._compute_current_waiting_time(sim.junction_id)
            reward = sim._compute_reward(current_waiting_time, step)
            done = sim._is_done(step)

            next_state = sim._get_state(sim.junction_id)
            next_states = np.vstack((sim.states, next_state))
            if len(next_states) > 4:
                # Delete first element
                next_states = np.delete(next_states, 0, 0)
            # Feed agent memory
            if len(sim.states) == 4:
                sim.agent.remember(sim.states, sim.action, reward, next_states, done)

            # Update
            sim.states = next_states
            sim.cumulative_reward += reward
            sim.cumulative_waiting_time += current_waiting_time
            sim.cumulative_intersection_queue += sim._compute_queue(sim.junction_id)

    sim.throughput += sim._compute_throughput()
    return sim


def extract_waiting_time(vehicle):
    # Return the waiting time only for the vehicle in queue
    if vehicle[1][122] > 0.5:
        return vehicle[1][122]
    return 0


def is_queued(vehicle):
    return vehicle[1][122] > 0.5


def compute_position_index(vehicle):
    # Initialize vehicle_position_index as -1 in order to not taking into account vehicles driving away from the tls later
    vehicle_position_index = -1
    # Lane position
    lane_pos = vehicle[1][86]
    # Lane ID
    lane_id = vehicle[1][81]
    # Lanes are 750m long, this calculate the distance from the tls
    distance_from_tls = 750 - lane_pos
    # Lane group initialization
    lane_group = -1
    # In which lane is the car? _3 are the "turn left only" lanes
    if '_3' in lane_id:
        if lane_id == 'W2TL_3':
            lane_group = 1
        elif lane_id == 'N2TL_3':
            lane_group = 3
        elif lane_id == 'E2TL_3':
            lane_group = 5
        elif lane_id == 'S2TL_3':
            lane_group = 7
    else:
        if 'W2TL' in lane_id:
            lane_group = 0
        elif 'N2TL' in lane_id:
            lane_group = 2
        elif 'E2TL' in lane_id:
            lane_group = 4
        elif 'S2TL' in lane_id:
            lane_group = 6

    if 0 <= lane_group <= 7:
        # distance in meters from the TLS -> mapping into cells
        if 400 <= distance_from_tls <= 750:
            lane_cell = 9
        elif 160 <= distance_from_tls < 400:
            lane_cell = 8
        elif 100 <= distance_from_tls < 160:
            lane_cell = 7
        elif 60 <= distance_from_tls < 100:
            lane_cell = 6
        elif 40 <= distance_from_tls < 60:
            lane_cell = 5
        elif 28 <= distance_from_tls < 40:
            lane_cell = 4
        elif 21 <= distance_from_tls < 28:
            lane_cell = 3
        elif 14 <= distance_from_tls < 21:
            lane_cell = 2
        elif 7 <= distance_from_tls < 14:
            lane_cell = 1
        elif distance_from_tls < 7:
            lane_cell = 0
        # composition of the two postion ID to create a number in the interval 0-79
        vehicle_position_index = int(str(lane_group) + str(lane_cell))

    return vehicle_position_index


def compute_car_state(vehicle):
    # Save the position of the vehicle as a point (vector) in a multi-dimensional space for faster computations
    position = np.zeros(80)
    # Save the speed of the vehicle as a point (vector) in a multi-dimensional space for faster computations
    speed = np.zeros(80)
    # Save the waiting_time of the vehicle as a point (vector) in a multi-dimensional space for faster computations
    waiting_time = np.zeros(80)
    # Save the queue_status of the vehicle as a point (vector) in a multi-dimensional space for faster computations
    queue_status = np.zeros(80)

    vehicle_position_index = compute_position_index(vehicle)

    # Do not consider vehicles driving away from the tls
    if vehicle_position_index > -1:
        position[vehicle_position_index] = 1
        speed[vehicle_position_index] = vehicle[1][64]
        if vehicle[1][122] > 0.5:
            queue_status[vehicle_position_index] = 1
            waiting_time[vehicle_position_index] = vehicle[1][122]

    return position, speed, waiting_time, queue_status


class Simulator:

    def __init__(self, label, sumocfg, state_size, max_steps, agent, gui=False):
        self.sumocfg = sumocfg
        self.state_size = state_size
        self.agent = agent
        self.max_steps = max_steps
        # stats
        self.cumulative_reward = 0
        self.cumulative_waiting_time = 0
        self.throughput = 0
        self.cumulative_intersection_queue = 0
        # tls state
        self.yellow_phase = False
        self.green_phase = False
        self.yellow_phase_step_count = 0
        self.green_phase_step_count = 0
        # action
        self.action = None
        self.previous_action = None

        self.label = label
        self.junction_id = 'TL'

        # Control SUMO mode (with or without GUI)
        if gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        # Start SUMO with TraCI and some flags
        traci.start([sumo_binary, "-c", self.sumocfg, "--no-step-log", "true"], label=self.label)

        self.connection = traci.getConnection(self.label)

        # The following code retrieves all vehicle speeds and waiting times within range (1000m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
        # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
        self.connection.junction.subscribeContext(self.junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000, [tc.VAR_SPEED, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID, tc.VAR_WAITING_TIME])
        # VAR_LANEPOSITION = 86
        # VAR_LANE_ID = 81
        # VAR_WAITING_TIME = 122
        # VAR_ARRIVED_VEHICLES_NUMBER = 121
        # VAR_SPEED = 64

        self.states = self._get_state(self.junction_id)

    def _compute_current_waiting_time(self, junction_id):
        subscription_results = self.connection.junction.getContextSubscriptionResults(junction_id)
        if subscription_results is not None:
            vehicles = subscription_results.items()
            return reduce(operator.add, map(extract_waiting_time, vehicles))
        return 0

    def _compute_reward(self, current_waiting_time, step):
        # Do not track first 100 steps
        if step < 100:
            return 0
        if current_waiting_time > 0:
            return 1 / current_waiting_time
        return 1

    def _compute_throughput(self):
        return self.connection.simulation.getArrivedNumber()

    def _compute_queue(self, junction_id):
        subscription_results = self.connection.junction.getContextSubscriptionResults(junction_id)
        queue = 0
        if subscription_results is not None:
            vehicles = subscription_results.items()
            queue = reduce(operator.add, map(is_queued, vehicles))
        return queue

    def _get_state(self, junction_id):
        subscription_results = self.connection.junction.getContextSubscriptionResults(junction_id)
        state = np.zeros(self.state_size)

        if subscription_results is not None:
            vehicles = subscription_results.items()
            # Filter out all the vehicles that are driving away from TLS
            vehicles = list(filter(lambda x: '2TL' in x[1][81], vehicles))

            cars_per_cell = np.zeros(80)
            avarage_speed_per_cell = np.zeros(80)
            cumulated_waiting_time_per_cell = np.zeros(80)
            queue_per_cell = np.zeros(80)

            if len(vehicles) > 0:
                positions, speeds, waiting_times, queue_statuses = zip(*map(compute_car_state, vehicles))

                # number of cars per cell going to the tls
                cars_per_cell = reduce(np.add, positions)

                # avarage speed per cell
                avarage_speed_per_cell = reduce(lambda x, y: np.mean([x, y], axis=0), positions)
                # cumulated waiting time per cell
                cumulated_waiting_time_per_cell = reduce(np.add, positions)
                # number of cars queued per cell
                queue_per_cell = reduce(np.add, queue_statuses)

            # tls phase
            tls_phase = np.array([self.connection.trafficlight.getPhase("TL")])

            state = np.concatenate([cars_per_cell, avarage_speed_per_cell, cumulated_waiting_time_per_cell, queue_per_cell, tls_phase])

        return state

    # Check if the episode is finished (not very useful here, but often used in Reinforcement Learning tasks)
    def _is_done(self, step):
        return step < self.max_steps - 1

    def do_step(self, step):
        # Choose self.action and (start yellow phase or execute self.action)
        if not self.green_phase and not self.yellow_phase:
            # Let the agent choose action
            # Store previous action
            self.previous_action = self.action
            # Choose action
            self.action = self.agent.act(self.states)
            # Start yellow phase
            if self.action != self.previous_action and self.previous_action is not None:
                self.yellow_phase = True
                # print('start yellow phase')
                # yellow phase number based on previous action
                self.connection.trafficlight.setPhase("TL", self.previous_action * 2 + 1)
            # Execute action
            else:
                # print('start green phase')
                self.green_phase = True
                if self.action == 0:
                    self.connection.trafficlight.setPhase("TL", PHASE_NS_GREEN)
                elif self.action == 1:
                    self.connection.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
                elif self.action == 2:
                    self.connection.trafficlight.setPhase("TL", PHASE_EW_GREEN)
                elif self.action == 3:
                    self.connection.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

        # Do step
        self.connection.simulationStep(float(step))

        # Update yellow phase counter
        if self.yellow_phase:
            # print('update yellow phase counter')
            self.yellow_phase_step_count += 1
            # Reset yellow phase
            if self.yellow_phase_step_count == YELLOW_PHASE_DURATION:
                # print('reset yellow phase count')
                self.yellow_phase = False
                self.yellow_phase_step_count = 0
        # Update green phase counter
        elif self.green_phase:
            # print('update green phase counter')
            self.green_phase_step_count += 1
            # Reset green phase
            if self.green_phase_step_count == GREEN_PHASE_DURATION:
                # print('reset green phase count')
                self.green_phase = False
                self.green_phase_step_count = 0
            elif self.green_phase_step_count == 1:
                # print('do things')
                current_waiting_time = self._compute_current_waiting_time(self.junction_id)
                reward = self._compute_reward(current_waiting_time, step)
                done = self._is_done(step)

                next_state = self._get_state(self.junction_id)
                next_states = np.vstack((self.states, next_state))
                if len(next_states) > 4:
                    # Delete first element
                    next_states = np.delete(next_states, 0, 0)
                # Feed agent memory
                if len(self.states) == 4:
                    self.agent.remember(self.states, self.action, reward, next_states, done)

                # Update
                self.states = next_states
                self.cumulative_reward += reward
                self.cumulative_waiting_time += current_waiting_time
                self.cumulative_intersection_queue += self._compute_queue(self.junction_id)

        self.throughput += self._compute_throughput()

    # end simulation
    def stop(self):
        self.connection.close(False)

        # Return the stats for this episode
        avg_waiting_time = self.cumulative_waiting_time / self.max_steps
        avg_intersection_queue = self.cumulative_intersection_queue / self.max_steps

        return self.cumulative_reward, avg_waiting_time, avg_intersection_queue, self.throughput
