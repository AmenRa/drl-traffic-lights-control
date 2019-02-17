from generate_routefile import generate_routefile
from simulator import Simulator
from agent import DQNAgent

# main entry point
if __name__ == "__main__":
    epochs = 100;
    max_time_steps = 1000
    # sumocfg = "environment/model.sumocfg"
    sumocfg = "environment/tlcs_config_test.sumocfg"
    # sumocfg = "environment/tlcs_config_train.sumocfg"
    tripinfo = "environment/tripinfo.xml"
    gui = False

    # Agent hyperparameters
    state_size
    action_size
    learning_rate = 0.0002
    gamma = 0.95
    epsilon = 1.0
    max_espilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    memory_size = 200
    name='DQNAgent'

    # Create DQNAgent
    agent = DQNAgent(
        state_size,
        action_size,
        learning_rate,
        gamma,
        epsilon,
        max_espilon,
        min_epsilon,
        decay_rate,
        memory_size,
        name
    )

    for epoch in epochs:
        # Generate routefile dynamically
        generate_routefile(max_time_steps)
        # Create Simulator
        sim = Simulator(sumocfg, tripinfo)
        # Run simulator
        sim.run(gui, max_time_steps)
