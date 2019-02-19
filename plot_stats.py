import matplotlib.pyplot as plt

def plot_stats(reward_store, avg_wait_store, throughput_store, avg_intersection_queue_store):

    plt.plot(reward_store) # reward
    plt.title("Reward")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")

    plt.plot(avg_wait_store) # average cumulative waiting time
    plt.title("Average cumulative delay")
    plt.ylabel("Average cumulative delay (s)")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")

    plt.plot(throughput_store) # num of cars arrived at the end
    plt.title("Intersection throughput")
    plt.ylabel("Throughput (vehicles)")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")

    plt.plot(avg_intersection_queue_store) # average number of cars in queue
    plt.title("Average intersection queue")
    plt.ylabel("Average queue length (vehicles)")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")
