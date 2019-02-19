import os
import sys
import numpy as np

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

from plot_stats import plot_stats

GREEN_PHASE_DURATION = 10 # duration of green phase
YELLOW_PHASE_DURATION = 4 # duration of yellow phase

# phase codes based on xai_tlcs.net.xml
PHASE_NS_GREEN = 0 # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2 # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4 # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6 # action 3 code 11
PHASE_EWL_YELLOW = 7

class Simulator:
    def __init__(self, sumocfg, tripinfo, state_size, agent):
        self.sumocfg = sumocfg
        self.tripinfo = tripinfo
        self.state_size = state_size
        self.agent = agent

    def get_state(self, junction_id):
        subscription_results = traci.junction.getContextSubscriptionResults(junction_id)
        total_waiting_time = 0
        state = np.zeros(self.state_size)
        intersection_queue = 0
        # number of cars arrived in the last step
        vehicles_arrived = traci.simulation.getArrivedNumber()

        if subscription_results is not None:
            for vehicle in subscription_results.items():
                total_waiting_time += vehicle[1][122]
                # heuristic to capture with precision when a car is really in queue
                if vehicle[1][122] > 0.5:
                    intersection_queue += 1
                lane_pos = vehicle[1][86]
                lane_id = vehicle[1][81]
                # inversion of lane so if it is close to TL, lane_pos = 0
                lane_pos = 750 - lane_pos
                # just dummy initialization
                lane_group = -1
                # flag for not detecting cars crossing the intersection or driving away from it
                valid_car = False

                # distance in meters from the TLS -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                # in which lane is the car? _3 are the "turn left only" lanes
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

                if 1 <= lane_group <= 7:
                    # composition of the two postion ID to create a number in interval 0-79
                    veh_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    veh_position = lane_cell
                    valid_car = True

                if valid_car:
                    # write the position of the car veh_id in the state array
                    state[veh_position] = 1

        # Transpose the state in order to feed it into the NN
        state = np.reshape(state, [1, self.state_size])
        return state, total_waiting_time, intersection_queue, vehicles_arrived

    def get_reward(self, previous_waiting_time, current_waiting_time):
        return previous_waiting_time - current_waiting_time

    def is_done(self, step, max_steps):
        return step < max_steps - 1

    # contains TraCI control loop
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

        state, previous_waiting_time, intersection_queue, vehicles_arrived = self.get_state(junction_id)

        yellow_phase = False
        green_phase = False
        yellow_phase_step_count = 0
        green_phase_step_count = 0

        action = None
        previous_action = None

        for step in range(max_steps):
            # Reset yellow phase
            if yellow_phase_step_count == YELLOW_PHASE_DURATION:
                yellow_phase = False
                yellow_phase_step_count = 0

            # Reset green phase
            if green_phase_step_count == GREEN_PHASE_DURATION:
                green_phase = False
                green_phase_step_count = 0

            # Let the agent change tls
            if not yellow_phase and not green_phase:
                # Store previous action
                previous_action = action
                # Choose action
                action = self.agent.act(state)
                if action == 0:
                    traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
                elif action == 1:
                    traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
                elif action == 2:
                    traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
                elif action == 3:
                    traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

            # Update yellow phase counter
            if yellow_phase:
                yellow_phase_step_count += 1
            # Update green phase counter
            if green_phase:
                green_phase_step_count += 1

            # Start yellow phase
            if action != previous_action and previous_action is not None:
                yellow_phase = True
                # yellow phase number based on previous action
                traci.trafficlight.setPhase("TL", previous_action * 2 + 1)
            # Start green phase
            else:
                green_phase = True

            traci.simulationStep(step)

            if green_phase == True and (green_phase_step_count in (0, GREEN_PHASE_DURATION)):
                next_state, current_waiting_time, intersection_queue, vehicles_arrived = self.get_state(junction_id)
                reward = self.get_reward(previous_waiting_time, current_waiting_time)
                done = self.is_done(step, max_steps)
                # Feed agent memory
                self.agent.remember(state, action, reward, next_state, done)

                # Update
                state = next_state
                previous_waiting_time = current_waiting_time
                cumulative_reward += reward
                cumulative_waiting_time += current_waiting_time
                throughput += vehicles_arrived
                cumulative_intersection_queue += intersection_queue

                # Train agent
                if len(self.agent.memory) >= 32:
                    self.agent.replay(batch_size)

        print("Total reward: {}, Eps: {}".format(cumulative_reward, self.agent.epsilon))

        traci.close()

        # Return the stats for this episode
        avg_waiting_time = cumulative_waiting_time / max_steps
        avg_intersection_queue = cumulative_intersection_queue / max_steps

        return cumulative_reward, avg_waiting_time, avg_intersection_queue, throughput
