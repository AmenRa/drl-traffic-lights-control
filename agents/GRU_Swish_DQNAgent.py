import os
import random
from collections import deque
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, CuDNNLSTM, Dropout, Add, Input, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

# Keras GPU utilization settings
import tensorflow as tf
# Avoid warning about tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# To log device placement (on which device the operation ran)
# config.log_device_placement = True
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)

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

    def __init__(self, state_size, action_size, memory_size=200, gamma=0.95, epsilon=1.0, epsilon_decay_rate=0.995, epsilon_min=0.01, learning_rate=0.0002, sample_size=32, batch_size=32, name='CuDNNLSTM_Swish_DQNAgent'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.name = name
        self.model = self._build_model()
        # self.graph = tf.get_default_graph()

    # Declare the model architecture and build the model
    def _build_model(self):
        def swish(x):
            return K.sigmoid(x) * x

        #
        # x1 = Dense(8, activation='relu')(input1)
        #
        # x2 = Dense(8, activation='relu')(input2)
        # # equivalent to added = add([x1, x2])
        # added = Add()([x1, x2])

        # Define inputs
        cells_input = Input(shape=(4, (self.state_size - 1)))
        lts_phase_input = Input((4, 1))

        # Define CuDNNLSTMs
        cells_LSTM = CuDNNLSTM(units=64)(cells_input)
        lts_phase_LSTM = CuDNNLSTM(units=1)(lts_phase_input)

        # Merge CuDNNLSTMs
        x = Add()([cells_LSTM, lts_phase_LSTM])

        # Define other layers
        # x = Dense(512, activation='relu')(x)
        x = Dense(256, activation=swish)(x)
        # x = Dropout(0.1)(x)
        x = Dense(256, activation=swish)(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation=swish)(x)
        # x = Dropout(0.1)(x)
        # x = Dense(64, activation='relu')(x)
        # x = Dropout(0.1)(x)
        # x = Dense(32, activation='relu')(x)
        output_layer = Dense(self.action_size, activation='tanh')(x)

        # Create model
        model = Model(inputs=[cells_input, lts_phase_input], outputs=output_layer)
        # model = Sequential()(merged_inputs)
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(self.action_size, activation='sigmoid'))

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
    def act(self, states):
        # np.random.rand() = random sample from a uniform distribution over [0, 1)
        if np.random.rand() <= self.epsilon or len(states) != 4:
            return random.randrange(self.action_size)
        # Transpose the states in order to feed it into the model
        states = states.reshape((1, 4, self.state_size))
        cells_states = states[:, :, :(self.state_size - 1)]
        tls_phase_states = states[:, :, [(self.state_size - 1)]]
        # K.set_learning_phase(0)
        # with self.graph.as_default():
        action_q_values = self.model.predict([cells_states, tls_phase_states])
        return np.argmax(action_q_values[0])

    def replay(self):
        # Sample from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Divide minibatch items' properties
        states = np.asarray([item[0] for item in minibatch])
        cells_states = states[:, :, :(self.state_size - 1)]
        tls_phase_states = states[:, :, [(self.state_size - 1)]]
        actions = np.asarray([item[1] for item in minibatch])
        rewards = np.asarray([item[2] for item in minibatch])
        next_states = np.asarray([item[3] for item in minibatch])
        cells_next_states = next_states[:, :, :(self.state_size - 1)]
        tls_phase_next_states = next_states[:, :, [(self.state_size - 1)]]

        # Predict the Q-value for each actions of each state
        # K.set_learning_phase(0)
        # with self.graph.as_default():
        currents_q_values = self.model.predict(x=[cells_states, tls_phase_states], batch_size=self.batch_size)
        # Predict the Q-value for each actions of each next state
        # K.set_learning_phase(0)
        # with self.graph.as_default():
        next_states_q_values = self.model.predict(x=[cells_next_states, tls_phase_next_states], batch_size=self.batch_size)

        # Update the Q-value of the choosen action for each state
        for current_q_values, action, reward, next_state_q_values in zip(currents_q_values, actions, rewards, next_states_q_values):
            # print(reward)
            # print(current_q_values[action])

            current_q_values[action] = reward + self.gamma * np.amax(next_state_q_values)

            if current_q_values[action] > 1:
                current_q_values[action] = 1
            # elif current_q_values[action] < 0:
            #     current_q_values[action] = 0
            elif current_q_values[action] < -1:
                current_q_values[action] = -1

            # print(current_q_values[action])
            # print('- - - - -')

        # import sys
        # sys.exit("Error message")

        # Train the model
        # K.set_learning_phase(1)
        # with self.graph.as_default():
        history = self.model.fit(
            x=[cells_states, tls_phase_states],
            y=currents_q_values,
            epochs=10,
            verbose=0,
            batch_size=self.batch_size
        )

        loss = history.history['loss']
        acc = history.history['acc']

        return loss, acc

        # Decrease epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay_rate

    # Load a pre-trained model
    def load(self):
        self.model.load_weights('agent_weights/' + self.name)

    # Load save the current model
    def save(self):
        self.model.save_weights('agent_weights/' + self.name)
