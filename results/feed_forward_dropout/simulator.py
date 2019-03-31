import os
import sys
import requests
from ast import literal_eval
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

    def __init__(self, label, sumocfg, state_size, max_steps, gui=False):
        self.sumocfg = sumocfg
        self.state_size = state_size
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
        self.current_waiting_time = 0

        # Control SUMO mode (with or without GUI)
        if gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        # Start SUMO with TraCI and some flags
        traci.start([sumo_binary, "-c", self.sumocfg, "--no-step-log", "true"], label=self.label)

        self.connection = traci.getConnection(self.label)

        # The following code retrieves all vehicle speeds and waiting times within range (1000m) of a junction (the vehicle ids are retrieved implicitly). The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.
        # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
        self.connection.junction.subscribeContext(self.junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000, [tc.VAR_SPEED, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID, tc.VAR_WAITING_TIME])
        # VAR_LANEPOSITION = 86
        # VAR_LANE_ID = 81
        # VAR_WAITING_TIME = 122
        # VAR_ARRIVED_VEHICLES_NUMBER = 121
        # VAR_SPEED = 64

        self.state = self._get_state(self.junction_id)

    def _compute_current_waiting_time(self, junction_id):
        subscription_results = self.connection.junction.getContextSubscriptionResults(junction_id)
        if subscription_results is not None:
            vehicles = subscription_results.items()
            return reduce(operator.add, map(extract_waiting_time, vehicles))
        return 0

    def _compute_reward(self, current_waiting_time, previous_waiting_time, step):
        return previous_waiting_time - current_waiting_time

    # def _compute_reward(self, current_waiting_time, current_queue, step):
    #     reward = 1
    #     if (current_queue > 0):
    #         avarage_current_waiting_time = current_waiting_time / current_queue
    #         if avarage_current_waiting_time > 20:
    #             reward = -1
    #         else:
    #             reward = (-avarage_current_waiting_time / 10) + 1
    #     return reward
    #
    # '''
    # DESMOS
    #
    # \left\{0\ \le x\ \le\ 10:-\left(ax^2+bx\ +\ c\right),x\ >\ 10:ax^2+bx\ +\ c\right\}
    #
    # a=-0.01
    # b=0.2
    # c=-1
    #
    # y\ =-\frac{1}{10}x\ +\ 1
    # '''

    # def _compute_reward(self, current_waiting_time, step):
    #     # Do not track first 100 steps
    #     if step < 100:
    #         return 0
    #     if current_waiting_time > 0:
    #         return 1 / current_waiting_time
    #     return 1

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
            cars_per_cell_normalized = np.zeros(80)
            avarage_speed_per_cell_normalized = np.zeros(80)
            cumulated_waiting_time_per_cell_normalized = np.zeros(80)
            queue_per_cell_normalized = np.zeros(80)

            if len(vehicles) > 0:
                positions, speeds, waiting_times, queue_statuses = zip(*map(compute_car_state, vehicles))

                # number of cars per cell going to the tls
                cars_per_cell = reduce(np.add, positions)
                if sum(cars_per_cell) > 0:
                    cars_per_cell_normalized = cars_per_cell / sum(cars_per_cell)
                # avarage speed per cell / ignore divide by zero warnings
                with np.errstate(divide='ignore', invalid='ignore'):
                    avarage_speed_per_cell = np.nan_to_num(np.divide(reduce(np.add, speeds), cars_per_cell))
                    avarage_speed_per_cell_normalized = avarage_speed_per_cell / 26
                # cumulated waiting time per cell
                cumulated_waiting_time_per_cell = reduce(np.add, waiting_times)
                if sum(cumulated_waiting_time_per_cell) > 0:
                    cumulated_waiting_time_per_cell_normalized = cumulated_waiting_time_per_cell / sum(cumulated_waiting_time_per_cell)
                # number of cars queued per cell
                queue_per_cell = reduce(np.add, queue_statuses)
                if sum(queue_per_cell) > 0:
                    queue_per_cell_normalized = queue_per_cell / sum(queue_per_cell)

            # tls phase
            tls_phase = np.array([self.connection.trafficlight.getPhase("TL")])

            state = np.concatenate([cars_per_cell_normalized, avarage_speed_per_cell_normalized, cumulated_waiting_time_per_cell_normalized, queue_per_cell_normalized])

            # state = np.concatenate([cars_per_cell, avarage_speed_per_cell, cumulated_waiting_time_per_cell, queue_per_cell, tls_phase])

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
            self.action = requests.post('http://127.0.0.1:5000/act', json={'states': self.state.tolist()}).json()['action']
            # Start yellow phase
            if self.action != self.previous_action and self.previous_action is not None:
                self.yellow_phase = True
                # Start yellow phase
                if self.action != self.previous_action and self.previous_action is not None:
                    self.yellow_phase = True
                    # yellow phase based on previous action
                    if self.previous_action == 0:
                        self.connection.trafficlight.setPhase('TL', PHASE_NS_YELLOW)
                    elif self.previous_action == 1:
                        self.connection.trafficlight.setPhase('TL', PHASE_NSL_YELLOW)
                    elif self.previous_action == 2:
                        self.connection.trafficlight.setPhase('TL', PHASE_EW_YELLOW)
                    elif self.previous_action == 3:
                        self.connection.trafficlight.setPhase('TL', PHASE_EWL_YELLOW)
            # Execute action
            else:
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

        # Compute stats
        current_queue = self._compute_queue(self.junction_id)
        previous_waiting_time = self.current_waiting_time
        self.current_waiting_time = self._compute_current_waiting_time(self.junction_id)
        reward = self._compute_reward(self.current_waiting_time, previous_waiting_time, step)

        # Update
        self.cumulative_reward += reward
        self.cumulative_waiting_time += self.current_waiting_time
        self.cumulative_intersection_queue += current_queue
        self.throughput += self._compute_throughput()

        done = self._is_done(step)

        # Update yellow phase counter
        if self.yellow_phase:
            # print('update yellow phase counter')
            self.yellow_phase_step_count += 1
            # Reset yellow phase
            if self.yellow_phase_step_count == YELLOW_PHASE_DURATION:
                # print('reset yellow phase count')
                self.yellow_phase = False
                self.yellow_phase_step_count = 0
                # Start green phase
                self.green_phase = True
                if self.action == 0:
                    self.connection.trafficlight.setPhase("TL", PHASE_NS_GREEN)
                elif self.action == 1:
                    self.connection.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
                elif self.action == 2:
                    self.connection.trafficlight.setPhase("TL", PHASE_EW_GREEN)
                elif self.action == 3:
                    self.connection.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
        # Update green phase counter
        elif self.green_phase:
            # print('update green phase counter')
            self.green_phase_step_count += 1
            # Reset green phase
            if self.green_phase_step_count == GREEN_PHASE_DURATION:
                # print('reset green phase count')
                self.green_phase = False
                self.green_phase_step_count = 0

                next_state = self._get_state(self.junction_id)
                # Feed agent memory
                requests.post('http://127.0.0.1:5000/remember', json={
                    'state': self.state.tolist(),
                    'action': self.action,
                    'reward': reward,
                    'next_state': next_state.tolist(),
                    'done': done
                })

                # Update state
                self.state = next_state

    # end simulation
    def stop(self):
        self.connection.close(False)

        # Return the stats for this episode
        avg_waiting_time = self.cumulative_waiting_time / self.max_steps
        avg_intersection_queue = self.cumulative_intersection_queue / self.max_steps

        return self.cumulative_reward, avg_waiting_time, avg_intersection_queue, self.throughput

    def run(self, max_steps):
        for step in range(0, max_steps):
            self.do_step(step)
        return self.stop()
