import matplotlib.pyplot as plt


saving_folder = 'output_figures/'


def plot_reward(agent_name, data, mode):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - ' + mode + ' - Reward')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episodes')
    # reward
    plt.plot(data)
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
    plt.savefig(saving_folder + agent_name + '-' + mode + '-reward.png', bbox_inches='tight', dpi=120)
    plt.close('all')


def plot_delay(agent_name, data, mode):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - ' + mode + ' - Average Cumulative Waiting Time per Step')
    plt.ylabel('Average Cumulative Waiting Time per Step (s)')
    plt.xlabel('Episodes')
    # average cumulative waiting time
    plt.plot(data)
    plt.savefig(saving_folder + agent_name + '-' + mode + '-average-cumulative-waiting-time.png', bbox_inches='tight')
    plt.close('all')


def plot_throughput(agent_name, data, mode):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - ' + mode + ' - Intersection Throughput')
    plt.ylabel('Throughput (vehicles)')
    plt.xlabel('Episodes')
    # num of cars arrived at the end
    plt.plot(data)
    plt.savefig(saving_folder + agent_name + '-' + mode + '-throughput.png', bbox_inches='tight')
    plt.close('all')


def plot_queue(agent_name, data, mode):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - ' + mode + ' - Average Queued Vehicles per Step')
    plt.ylabel('Average queue length (vehicles)')
    plt.xlabel('Episodes')
    # average number of cars in queue
    plt.plot(data)
    plt.savefig(saving_folder + agent_name + '-' + mode + '-average-queued-vehicles-per-step.png', bbox_inches='tight')
    plt.close('all')


def plot_rewards(agent_name, data):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - Reward')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episodes')
    # reward
    plt.plot(data[0::4], label='Low')
    plt.plot(data[1::4], label='High')
    plt.plot(data[2::4], label='North-South')
    plt.plot(data[3::4], label='East-West')
    plt.legend()
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
    plt.savefig(saving_folder + agent_name + '-rewards.png', bbox_inches='tight', dpi=120)
    plt.close('all')


def plot_delays(agent_name, data):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - Average Cumulative Waiting Time per Step')
    plt.ylabel('Average Cumulative Waiting Time per Step (s)')
    plt.xlabel('Episodes')
    # reward
    plt.plot(data[0::4], label='Low')
    plt.plot(data[1::4], label='High')
    plt.plot(data[2::4], label='North-South')
    plt.plot(data[3::4], label='East-West')
    plt.legend()
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
    plt.savefig(saving_folder + agent_name + '-average-cumulative-waiting-times.png', bbox_inches='tight', dpi=120)
    plt.close('all')


def plot_throughputs(agent_name, data):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - Intersection Throughput')
    plt.ylabel('Throughput (vehicles)')
    plt.xlabel('Episodes')
    # reward
    plt.plot(data[0::4], label='Low')
    plt.plot(data[1::4], label='High')
    plt.plot(data[2::4], label='North-South')
    plt.plot(data[3::4], label='East-West')
    plt.legend()
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
    plt.savefig(saving_folder + agent_name + '-throughputs.png', bbox_inches='tight', dpi=120)
    plt.close('all')


def plot_queues(agent_name, data):
    plt.figure(figsize=(10, 5), dpi=120)
    plt.title(agent_name + ' - Average Queued Vehicles per Step')
    plt.ylabel('Average queue length (vehicles)')
    plt.xlabel('Episodes')
    # reward
    plt.plot(data[0::4], label='Low')
    plt.plot(data[1::4], label='High')
    plt.plot(data[2::4], label='North-South')
    plt.plot(data[3::4], label='East-West')
    plt.legend()
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
    plt.savefig(saving_folder + agent_name + '-average-queued-vehicles-per-steps.png', bbox_inches='tight', dpi=120)
    plt.close('all')


def plot_stats(agent_name, reward_store, avg_wait_store, throughput_store, avg_intersection_queue_store):

    avg_wait_store = [x * 3600 / 150 for x in avg_wait_store]
    avg_intersection_queue_store = [x * 3600 / 150 for x in avg_intersection_queue_store]

    plot_reward(agent_name, reward_store[0::4], 'Low')
    plot_reward(agent_name, reward_store[1::4], 'High')
    plot_reward(agent_name, reward_store[2::4], 'North-South')
    plot_reward(agent_name, reward_store[3::4], 'East-West')

    plot_delay(agent_name, avg_wait_store[0::4], 'Low')
    plot_delay(agent_name, avg_wait_store[1::4], 'High')
    plot_delay(agent_name, avg_wait_store[2::4], 'North-South')
    plot_delay(agent_name, avg_wait_store[3::4], 'East-West')

    plot_throughput(agent_name, throughput_store[0::4], 'Low')
    plot_throughput(agent_name, throughput_store[1::4], 'High')
    plot_throughput(agent_name, throughput_store[2::4], 'North-South')
    plot_throughput(agent_name, throughput_store[3::4], 'East-West')

    plot_queue(agent_name, avg_intersection_queue_store[0::4], 'Low')
    plot_queue(agent_name, avg_intersection_queue_store[1::4], 'High')
    plot_queue(agent_name, avg_intersection_queue_store[2::4], 'North-South')
    plot_queue(agent_name, avg_intersection_queue_store[3::4], 'East-West')

    plot_rewards(agent_name, reward_store)
    plot_delays(agent_name, avg_wait_store)
    plot_throughputs(agent_name, throughput_store)
    plot_queues(agent_name, avg_intersection_queue_store)
