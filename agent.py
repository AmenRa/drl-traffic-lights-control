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

        self.memory = deque(maxlen=memory_size) # Number of experiences the Memory can keep

        self.optimizer
        self.loss_function
        self.variable_initializer

        self.model = self.build_model()

        # # Exploration parameters for epsilon greedy strategy
        # # exploration probability at start
        # self.explore_start = 1.0
        # # minimum exploration probability
        # self.explore_stop = 0.01
        # # exponential decay rate for exploration prob
        # self.decay_rate = 0.0001

    def build_model(self):
        """
        NETWORK ARCHITECTURE GOES HERE
        """

    def train_batch(self, ...):
        """
        TRAIN STEP
        """

    def act(self, state):
        """
        RETURN ACTION
        """

    def replay(self, batch_size):
        """
        REPLAY IMPLEMENTATION GOES HERE
        """

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
