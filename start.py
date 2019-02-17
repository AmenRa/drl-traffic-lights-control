from generate_routefile import generate_routefile
from simulator import Simulator

# main entry point
if __name__ == "__main__":
    epochs = 100;
    max_time_steps = 1000
    # sumocfg = "environment/model.sumocfg"
    sumocfg = "environment/tlcs_config_test.sumocfg"
    # sumocfg = "environment/tlcs_config_train.sumocfg"
    tripinfo = "environment/tripinfo.xml"
    gui = False

    # DL AGENT PARAMETERS
    # state_size
    # action_size
    # learning_rate = 0.0002
    # gamma = 0.95
    # epsilon = 0.1
    # memory_size = 200

    # TLS PHASES

    # Generate routefile dynamically
    generate_routefile(max_time_steps)
    # Create Simulator
    sim = Simulator(sumocfg, tripinfo)
    # Run simulator
    sim.run(gui, max_time_steps)
