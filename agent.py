import random
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

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
        self.model = self._build_model()

    # Declare the model architecture and build the model
    def _build_model(self):
        # model = Sequential()
        #
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense24, activation='relu')
        # model.add(Dense(self.action_size, activation='linear'))
        #
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # Create a sequntial model (a sequential model is a linear stack of layers)
        model = Sequential([
            Dense(1000, input_dim=self.state_size),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
            Dense(1000),
            Activation('relu'),
        ])

        sgd = optimizers.SGD(lr=self.learning_rate)

        model.compile(
            optimizer=sgd,
            loss='mse',
            metrics=['accuracy']
        )

        return model

    # Choose an action
    def act(self, state):
        # np.random.rand() = random sample from a uniform distribution over [0, 1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_q_values = self.model.predict(state)
        return np.argmax(action_q_values[0])

    # Save a sample into memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # The learning happens here
    def replay(self, batch_size):
        # Sample from memory
        minibatch = random.sample(self.memory, batch_size)

        # Divide minibatch items' properties
        states = [item[0] for item in minibatch]
        actions = [item[1] for item in minibatch]
        next_states = [item[2] for item in minibatch]
        rewards = [item[3] for item in minibatch]

        # Predict the Q-value for each actions of each state
        currents_q_values = self.model.predict(x = states, batch_size)
        # Predict the Q-value for each actions of each next state
        next_states_q_values = self.model.predict(x = next_states, batch_size)

        # Update the Q-value of the choosen action for each state
        for current_q_values, action, reward, next_state_q_values in zip(currents_q_values, actions, rewards, next_states_q_values):
            current_q_values[action] = reward + self.gamma * np.amax(next_state_q_values)

        # Set input and output data
        x = np.array([state for state in states])
        y = currents_q_values

        # Train the NN
        self.model.fit(x, y, epochs=1, verbose=0)

    # Load a pre-trained model
    def load(self, name):
        self.model.load_weights(name)

    # Load save the current model
    def save(self, name):
        self.model.save_weights(name)
