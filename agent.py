import random
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate = 0.0002, gamma = 0.95, epsilon = 0.1, memory_size = 200, name='DQNAgent'):
        # Model hyperparameters
        self.state_size = state_size
        self.action_size = action_size

        # Training hyperparameters
        self.learning_rate = learning_rate

        # Q learning hyperparameters
        # discount rate
        self.gamma = gamma
        # exploration rate
        self.epsilon = epsilon

        # Number of experiences the Memory can keep
        self.memory = deque(maxlen=memory_size)

        self.optimizer
        self.loss_function
        self.variable_initializer

        self.model = self.build_model()

    def build_model(self):
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

    def act(self, state):
        # np.random.rand() = random sample from a uniform distribution over [0, 1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
