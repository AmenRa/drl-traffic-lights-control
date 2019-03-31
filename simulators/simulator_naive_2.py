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


class Simulator:

    def __init__(self, label, sumocfg, max_steps, gui=False):
        self.sumocfg = sumocfg
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

    def _compute_current_waiting_time(self, junction_id):
        subscription_results = self.connection.junction.getContextSubscriptionResults(junction_id)
        if subscription_results is not None:
            vehicles = subscription_results.items()
            return reduce(operator.add, map(extract_waiting_time, vehicles))
        return 0

    def _compute_reward(self, current_waiting_time, previous_waiting_time, step):
        return previous_waiting_time - current_waiting_time

    def _compute_throughput(self):
        return self.connection.simulation.getArrivedNumber()

    def _compute_queue(self, junction_id):
        subscription_results = self.connection.junction.getContextSubscriptionResults(junction_id)
        queue = 0
        if subscription_results is not None:
            vehicles = subscription_results.items()
            queue = reduce(operator.add, map(is_queued, vehicles))
        return queue

    def do_step(self, step):
        # Choose self.action and (start yellow phase or execute self.action)
        if (not self.green_phase and not self.yellow_phase):
            # Let the agent choose action
            # Store previous action
            self.previous_action = self.action
            # Choose action
            if self.previous_action in [0, 1, 2]:
                self.action = self.previous_action + 1
            # elif self.previous_action == 3:
            #     self.action = 0
            else:
                self.action = 0

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
