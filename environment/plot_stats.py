import matplotlib.pyplot as plt

def plot_stats(gr):

    plt.plot(gr.reward_store) # reward
    plt.title("Reward")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")

    plt.plot(gr.avg_wait_store) # average cumulative wait time
    plt.title("Average cumulative delay")
    plt.ylabel("Average cumulative delay (s)")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")

    plt.plot(gr.throughput_store) # num of cars arrived at the end
    plt.title("Intersection throughput")
    plt.ylabel("Throughput (vehicles)")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")

    plt.plot(gr.avg_intersection_queue_store) # average number of cars in queue
    plt.title("Average intersection queue")
    plt.ylabel("Average queue length (vehicles)")
    plt.xlabel("Epoch")
    plt.show()
    plt.close("all")
