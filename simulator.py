import os
import sys
import numpy as np
import time

# Import some Python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import optparse
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import traci.constants as tc

class SubscriptionListener(traci.StepListener):
    def __init__(self, junctionID):
        self.junctionID = junctionID

    # do all the elaboration from raw SUMO data (MOVE IT IN ANOTHER FILE)
    # def represent_state(self):

    # Get SUMO state
    def get_simulator_state(self):
        vehicle_ids = traci.vehicle.getIDList()
        # function_to_apply list_of_inputs
        lane_positions = map(lambda id: traci.vehicle.getLanePosition(id), vehicle_ids)


    def step(self, t=0):
        # do something at every simulaton step
        # print(traci.junction.getContextSubscriptionResults(self.junctionID))

        vehicle_ids = traci.vehicle.getIDList()

        # Needed to retrieve lane_ids of vehicles
        lane_positions = list(map(lambda veh_id: traci.vehicle.getLanePosition(veh_id), vehicle_ids))

        # From lane ids we can give a value to direction
        lane_ids = list(map(lambda veh_id: traci.vehicle.getLaneID(veh_id), vehicle_ids))

        # Acumulated waiting times
        accumulatedWaitingTimes = list(map(lambda veh_id: traci.vehicle.getAccumulatedWaitingTime(veh_id), vehicle_ids))

        # Speeds
        vehicle_speeds = list(map(lambda veh_id: traci.vehicle.getSpeed(veh_id), vehicle_ids))

        tls_id = traci.trafficlight.getIDList()[0]

        # ACTION
        # traci.trafficlight.setPhase(tls_id, 0)

        # if t%2 == 0:
        #     traci.trafficlight.setPhase(tls_id, 0)
        # else:
        #     traci.trafficlight.setPhase(tls_id, 1)


        # REWARD SHOULD BE CALCULATED HERE
        # It is equal to the cumulative waiting time at each step
        not_moving_vechicle_count = list(filter(lambda x: x < 0.1, vehicle_speeds))
        reward = -len(not_moving_vechicle_count)
        print(reward)

        # indicate that the step listener should stay active in the next step
        return True

class Simulator:
    def __init__(self, sumocfg, tripinfo):
        self.sumocfg = sumocfg
        self.tripinfo = tripinfo

    # contains TraCI control loop
    def run(self, gui = False, max_time_steps = 100):
        start_time = time.time()

        # Control SUMO mode (with or without GUI)
        if gui == False:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # Start SUMO with TraCI and some flags
        traci.start([sumoBinary, "-c", self.sumocfg, "--tripinfo-output", self.tripinfo])

        junctionID = "1_1_0"

        # The following code retrieves all vehicle speeds and waiting times within range (50m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
        # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
        traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])

        gneJ4_listener = SubscriptionListener(junctionID)
        traci.addStepListener(gneJ4_listener)

        for step in range(max_time_steps):
           print("Step: ", step)
           traci.simulationStep(step)

        traci.close()

        elapsed_time = round(time.time() - start_time, 2)

        print("----- Elapsed time: ", elapsed_time, " seconds -----")

        sys.stdout.flush()
