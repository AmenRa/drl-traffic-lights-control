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
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7


class Simulator:

    def __init__(self, sumocfg, state_size, agent):
        self.sumocfg = sumocfg
        self.state_size = state_size
        self.agent = agent

    def _compute_current_waiting_time(self, junction_id):
        subscription_results = traci.junction.getContextSubscriptionResults(junction_id)
        if subscription_results is not None:
            vehicles = subscription_results.items()
            return reduce(operator.add, map(lambda x: x[1][122], vehicles))
        return 0

    def _compute_reward(self, current_waiting_time, step):
        # Do not track first 100 steps
        if step < 100:
            return 0
        if current_waiting_time > 0:
            return 1 / current_waiting_time
        return 1

    def _compute_throughput(self):
        return traci.simulation.getArrivedNumber()

    def _compute_queue(self, junction_id):
        subscription_results = traci.junction.getContextSubscriptionResults(junction_id)
        queue = 0
        if subscription_results is not None:
            vehicles = subscription_results.items()
            queue = reduce(operator.add, map(lambda x: x[1][122] > 0.5, vehicles))
        return queue

    def _compute_position_index(self, vehicle):
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

        if 0 <= lane_group <= 7:
            # composition of the two postion ID to create a number in the interval 0-79
            vehicle_position_index = int(str(lane_group) + str(lane_cell))

        return vehicle_position_index

    def _compute_car_state(self, vehicle):
        # Save the position of the vehicle as a point (vector) in a multi-dimensional space for faster computations
        position = np.zeros(80)
        # Save the speed of the vehicle as a point (vector) in a multi-dimensional space for faster computations
        speed = np.zeros(80)
        # Save the waiting_time of the vehicle as a point (vector) in a multi-dimensional space for faster computations
        waiting_time = np.zeros(80)
        # Save the queue_status of the vehicle as a point (vector) in a multi-dimensional space for faster computations
        queue_status = np.zeros(80)

        vehicle_position_index = self._compute_position_index(vehicle)

        # Do not consider vehicles driving away from the tls
        if vehicle_position_index > -1:
            position[vehicle_position_index] = 1
            speed[vehicle_position_index] = vehicle[1][64]
            waiting_time[vehicle_position_index] = vehicle[1][122]
            if vehicle[1][122] > 0.5:
                queue_status[vehicle_position_index] = 1

        return position, speed, waiting_time, queue_status

    def _get_state(self, junction_id):
        subscription_results = traci.junction.getContextSubscriptionResults(junction_id)
        state = np.zeros(self.state_size)

        if subscription_results is not None:
            vehicles = subscription_results.items()

            positions, speeds, waiting_times, queue_statuses = zip(*map(self._compute_car_state, vehicles))

            # number of cars per cell going to the tls
            cars_per_cell = reduce(np.add, positions)
            # avarage speed per cell
            avarage_speed_per_cell = reduce(lambda x, y: np.mean([x, y], axis=0), positions)
            # cumulated waiting time per cell
            cumulated_waiting_time_per_cell = reduce(np.add, positions)
            # number of cars queued per cell
            queue_per_cell = reduce(np.add, queue_statuses)
            # tls phase
            tls_phase = np.array([traci.trafficlight.getPhase("TL")])

            state = np.concatenate([cars_per_cell, avarage_speed_per_cell, cumulated_waiting_time_per_cell, queue_per_cell, tls_phase])

        return state

    # Check if the episode is finished (not very useful here, but often used in Reinforcement Learning tasks)
    def _is_done(self, step, max_steps):
        return step < max_steps - 1

    # Run simulation (TraCI/SUMO)
    def run(self, gui=False, max_steps=100, batch_size=32):
        # Control SUMO mode (with or without GUI)
        if gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        # Start SUMO with TraCI and some flags
        traci.start([sumo_binary, "-c", self.sumocfg, "--no-step-log", "true"])

        junction_id = "TL"

        # The following code retrieves all vehicle speeds and waiting times within range (50m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
        # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
        traci.junction.subscribeContext(junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000, [tc.VAR_SPEED, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID, tc.VAR_WAITING_TIME])
        # VAR_LANEPOSITION = 86
        # VAR_LANE_ID = 81
        # VAR_WAITING_TIME = 122
        # VAR_ARRIVED_VEHICLES_NUMBER = 121
        # VAR_SPEED = 64

        # stats
        cumulative_reward = 0
        cumulative_waiting_time = 0
        throughput = 0
        cumulative_intersection_queue = 0

        state = self._get_state(junction_id)

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
                    next_state = self._get_state(junction_id)
                    current_waiting_time = self._compute_current_waiting_time(junction_id)
                    reward = self._compute_reward(current_waiting_time, step)
                    done = self._is_done(step, max_steps)
                    # Feed agent memory
                    self.agent.remember(state, action, reward, next_state, done)
                    # Update
                    state = next_state
                    cumulative_reward += reward
                    cumulative_waiting_time += current_waiting_time
                    cumulative_intersection_queue += self._compute_queue(junction_id)

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
