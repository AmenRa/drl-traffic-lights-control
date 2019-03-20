import matplotlib.pyplot as plt


def plot_stats(reward_store, avg_wait_store, throughput_store, avg_intersection_queue_store):

    plt.figure(figsize=(10, 5), dpi=120)
    plt.title("Reward")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Epoch")
    # reward
    plt.plot(reward_store)
    # plt.show()
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
    plt.savefig('output_figures/reward.png', bbox_inches='tight', dpi=120)
    plt.close("all")

    # average cumulative waiting time
    plt.plot(avg_wait_store)
    plt.title("Average cumulative delay")
    plt.ylabel("Average cumulative delay (s)")
    plt.xlabel("Epoch")
    # plt.show()
    plt.savefig('output_figures/avarage-cumulative-delay.png', bbox_inches='tight')
    plt.close("all")

    # num of cars arrived at the end
    plt.plot(throughput_store)
    plt.title("Intersection throughput")
    plt.ylabel("Throughput (vehicles)")
    plt.xlabel("Epoch")
    # plt.show()
    plt.savefig('output_figures/throughput.png', bbox_inches='tight')
    plt.close("all")

    # average number of cars in queue
    plt.plot(avg_intersection_queue_store)
    plt.title("Average intersection queue")
    plt.ylabel("Average queue length (vehicles)")
    plt.xlabel("Epoch")
    # plt.show()
    plt.savefig('output_figures/avarage-queue.png', bbox_inches='tight')
    plt.close("all")
