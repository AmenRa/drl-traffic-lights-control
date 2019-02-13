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

    # step = 0
    # # while traci.simulation.getMinExpectedNumber() > 0:
    # while step < 1:
    #     # Forces SUMO to perform simulation.
    #     traci.simulationStep()
    #     print(step)
    #     print(traci.getAllSubscriptionResults())
    #     step += 1

    junctionID = "gneJ4"

    # The following code retrieves all vehicle speeds and waiting times within range (50m) of a junction (the vehicle ids are retrieved implicitly). (The values retrieved are always the ones from the last time step, it is not possible to retrieve older values.)
    traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_SPEED, tc.VAR_WAITING_TIME, tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_POSITION])

    gneJ4_listener = SubscriptionListener(junctionID)
    traci.addStepListener(gneJ4_listener)

    for step in range(100):
       print("step", step)
       traci.simulationStep()

    traci.close()
    sys.stdout.flush()


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([sumoBinary, "-c", "test_environment/demo.sumocfg",
                             "--tripinfo-output", "test_environment/tripinfo.xml"])
    run()
