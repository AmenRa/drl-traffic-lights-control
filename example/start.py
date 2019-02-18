output_dir = 'model_output'
done = False

for e in range (n_episodes):
    state = get_state()
    for step in n_steps:
        action = agent.act(state)
        do_step()
        next_state = get_state()
        reward = get_reward()
        done = is_done()

        # As long as the simulation is not finished or the goal reached, reward does not change
        reward = reward if not done else penalty

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("episodes: {}/{}, score: {}, epsilon: {:2}".format(e, n_episodes, throughput, agent.epsilon))
            break

    # Train agent
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Save model weights every 50 episodes
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}').format(e) + ".hdf5")
