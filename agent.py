import random
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

'''
state_size                  Size of the input
action_size                 Size of the output
memory_size                 Size of the memory
gamma                       Discounting reward rate
epsilon                     Exploration rate
epsilon_decay_rate          Decay rate for exploration probability
epsilon_min                 Minimum exploration probability
learning_rate               Learning rate
name                        The name used to save the model

# Reduce epsilon (because we need less and less exploration)
epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_decay_rate*episode)
'''

class DQNAgent:

    def __init__(self, state_size, action_size, memory_size=200, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.0002, name='DQNAgent'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.name = name
        self.model = self._build_model()

    # Declare the model architecture and build the model
    def _build_model(self):
        # Create a sequntial model (a sequential model is a linear stack of layers)
        model = Sequential()

        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))

        model.compile(
            # optimizer=SGD(lr=self.learning_rate),
            optimizer=Adam(lr=self.learning_rate),
            loss='mse',
            metrics=['accuracy']
        )

        return model

    # Save a sample into memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action
    def act(self, state):
        # np.random.rand() = random sample from a uniform distribution over [0, 1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Transpose the state in order to feed it into the model
        state = np.reshape(state, [1, self.state_size])
        action_q_values = self.model.predict(state)
        return np.argmax(action_q_values[0])

    # The learning happens here
    def old_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Transpose the state in order to feed it into the model
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(state, [1, self.state_size])
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, batch_size=1)[0])
            target_f = self.model.predict(state, batch_size=1)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate

    def replay(self, batch_size):
        # Sample from memory
        minibatch = random.sample(self.memory, batch_size)

        # Divide minibatch items' properties
        states = np.asarray([item[0] for item in minibatch])
        actions = np.asarray([item[1] for item in minibatch])
        rewards = np.asarray([item[2] for item in minibatch])
        next_states = np.asarray([item[3] for item in minibatch])

        # Predict the Q-value for each actions of each state
        currents_q_values = self.model.predict(x=states, batch_size=batch_size)
        # Predict the Q-value for each actions of each next state
        next_states_q_values = self.model.predict(x=next_states, batch_size=batch_size)

        # Update the Q-value of the choosen action for each state
        for current_q_values, action, reward, next_state_q_values in zip(currents_q_values, actions, rewards, next_states_q_values):
            current_q_values[action] = reward + self.gamma * np.amax(next_state_q_values)

        # Train the model
        self.model.fit(
            x=states,
            y=currents_q_values,
            epochs=1,
            verbose=0
        )

        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate

    # Load a pre-trained model
    def load(self):
        self.model.load_weights(self.name)

    # Load save the current model
    def save(self):
        self.model.save_weights(self.name)
