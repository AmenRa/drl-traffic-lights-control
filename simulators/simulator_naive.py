import os
import sys
# This module exports a set of functions implemented in C corresponding to the intrinsic operators of Python. For example, operator.add(x, y) is equivalent to the expression x+y. The function names are those used for special methods; variants without leading and trailing '__' are also provided for convenience.
import operator
from functools import reduce

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

    def __init__(self, sumocfg):
        self.sumocfg = sumocfg

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

        # The following code retrieves all vehicle speeds and waiting times within range (1000m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
        # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
        traci.junction.subscribeContext(junction_id, tc.CMD_GET_VEHICLE_VARIABLE, 1000, [tc.VAR_SPEED, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID, tc.VAR_WAITING_TIME])
        # VAR_LANEPOSITION = 86
        # VAR_LANE_ID = 81
        # VAR_WAITING_TIME = 122
        # VAR_ARRIVED_VEHICLES_NUMBER = 121
        # VAR_SPEED = 64

        # Initialize metrics
        cumulative_waiting_time = 0
        throughput = 0
        cumulative_intersection_queue = 0

        for step in range(max_steps):
            # Do step
            traci.simulationStep(step)
            # Update metrics
            throughput += self._compute_throughput()
            cumulative_intersection_queue += self._compute_queue(junction_id)
            cumulative_waiting_time += self._compute_current_waiting_time(junction_id)

        traci.close(False)

        # Return the stats for this episode
        avg_waiting_time = cumulative_waiting_time / max_steps
        avg_intersection_queue = cumulative_intersection_queue / max_steps

        return avg_waiting_time, avg_intersection_queue, throughput
