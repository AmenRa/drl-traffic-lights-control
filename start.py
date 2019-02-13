import os
import sys
import optparse

# we need to import some python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


from sumolib import checkBinary  # Checks for the binary in environ vars
import traci

# from geneterate_env import GenerateEnv

from generate_routefile import generate_routefile

import traci.constants as tc

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options



class SubscriptionListener(traci.StepListener):
    def __init__(self, junctionID):
        self.junctionID = junctionID

    def step(self, t=0):
        # do something at every simulaton step
        print(traci.junction.getContextSubscriptionResults(self.junctionID))
        # indicate that the step listener should stay active in the next step
        return True

# contains TraCI control loop
def run():

    junctionID = "1_1_0"

    # The following code retrieves all vehicle speeds and waiting times within range (50m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
    # add tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION if more than one tl
    traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])

    # gneJ4_listener = SubscriptionListener(junctionID)
    # traci.addStepListener(gneJ4_listener)

    for step in range(100):
       print("Step: ", step)
       traci.simulationStep()

    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    # # Generate env
    # generateEnv = GenerateEnv()
    # generateEnv.generate_routefile()

    # Generate routefile dynamically
    generate_routefile()

    # this function is called from the command line. It will start sumo as a server, then connect and run
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "environment/model.sumocfg",
                             "--tripinfo-output", "environment/tripinfo.xml"])
    run()
