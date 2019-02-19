class SimRunner:
    def __init__(self, sess, model, memory):
        self._sess = sess
        self._model = model
        self._memory = memory
        self._eps = 0
        self._steps = 0
        self._reward_store = []
        self._throughput_store = []
        self._avg_wait_store = []
        self._avg_intersection_queue_store = []

    def run(self, epoch, sumoCmd, gather_stats):

        # set the epsilon for this episode
        self._eps = 1.0 - (epoch / TOTAL_EPOCHS)

        self._steps = 0
        tot_reward = 0
        old_wait_time = 0
        sum_wait_time = 0
        sum_car_in_queue = 0
        sum_intersection_queue = 0
        throughput = 0
        cumulative_wait = 0

        while self._steps < MAX_STEPS:
            #  calculate reward: (change in cumulative reward between actions)
            #current_wait_time = self._get_wait_time()
            current_wait_time = cumulative_wait # this is the situation after the action (last scan of the intersection)
            reward = old_wait_time - current_wait_time

            # get current state of the intersection
            current_state = self._get_state()

            # data saving into memory & training - if the sim is just started, there is no old_state
            if self._steps != 0:
                self._memory.add_sample((old_state, old_action, reward, current_state))
                self._replay()

            # choose the action to perform based on the current state
            action = self._choose_action(current_state)

            # if the chosen action is different from the last one, its time for the yellow phase
            if self._steps != 0 and old_action != action: # dont do this in the first step, old_action doesnt exists
                self._set_yellow_phase(old_action)
                steps_todo = self._calculate_steps(YELLOW_PHASE_SEC)
                self._steps = self._steps + steps_todo
                while steps_todo > 0:
                    traci.simulationStep()
                    steps_todo -= 1
                    if gather_stats == True: # really precise stats but very SLOW to compute
                        intersection_queue, cumulative_wait, arrived_now = self._get_stats()
                        sum_intersection_queue += intersection_queue
                        sum_wait_time += cumulative_wait
                        throughput += arrived_now

            # execute the action selected before
            self._execute_action(action)
            steps_todo = self._calculate_steps(GREEN_PHASE_SEC)
            self._steps = self._steps + steps_todo
            while steps_todo > 0:
                traci.simulationStep()
                steps_todo -= 1
                if gather_stats == True: # really precise stats but very SLOW to compute
                    intersection_queue, cumulative_wait, arrived_now = self._get_stats()
                    sum_intersection_queue += intersection_queue
                    sum_wait_time += cumulative_wait
                    throughput += arrived_now

            if gather_stats == False: # not so precise stats, but fast computation
                intersection_queue, cumulative_wait, arrived_now = self._get_stats()
                sum_intersection_queue += intersection_queue
                sum_wait_time += cumulative_wait
                throughput += arrived_now

            # saving the variables for the next step & accumulate reward
            old_state = current_state
            old_action = action
            old_wait_time = current_wait_time
            tot_reward += reward

        # save the stats for this episode
        self._reward_store.append(tot_reward)
        self._avg_wait_store.append(sum_wait_time / MAX_STEPS)
        self._throughput_store.append(throughput)
        self._avg_intersection_queue_store.append(sum_intersection_queue / MAX_STEPS)

        print("Total reward: {}, Eps: {}".format(tot_reward, self._eps))
        traci.close()

    # def _choose_action(self, state):
    #     if random.random() < self._eps: # epsilon controls the randomness of the action
    #         return random.randint(0, self._model.num_actions - 1) # random action
    #     else:
    #         return np.argmax(self._model.predict_one(state, self._sess)) # the best action given the current state (prediction from nn)
    #
    # def _set_yellow_phase(self, old_action):
    #     yellow_phase = old_action * 2 + 1 # obtain the correct yellow phase number based on the old action
    #     traci.trafficlight.setPhase("TL", yellow_phase)
    #
    # def _execute_action(self, action_number):
    #     if action_number == 0:
    #         traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
    #     elif action_number == 1:
    #         traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
    #     elif action_number == 2:
    #         traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
    #     elif action_number == 3:
    #         traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
    #
    # def _calculate_steps(self, phase_steps):
    #     if (self._steps + phase_steps) >= MAX_STEPS: # check if the steps to do are over the limit of MAX_STEPS
    #         phase_steps = MAX_STEPS - self._steps
    #     return phase_steps

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def avg_wait_store(self):
        return self._avg_wait_store

    @property
    def throughput_store(self):
        return self._throughput_store

    @property
    def avg_intersection_queue_store(self):
        return self._avg_intersection_queue_store
