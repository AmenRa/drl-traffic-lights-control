#!/usr/bin/python3
from flask import Flask, request, jsonify
import numpy as np
import time
from DQNAgent import DQNAgent

# Disable non-error logs
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Agent hyperparameters
STATE_SIZE = 320
ACTION_SIZE = 4
MEMORY_SIZE = 1024
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY_RATE = 0.99999
EPSILON_MIN = 0.01
LEARNING_RATE = 0.0002
SAMPLE_SIZE = 128
BATCH_SIZE = 32
NAME = 'ffdo_DQNAgent'

# Create DQNAgent
DQNAgent = DQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    memory_size=MEMORY_SIZE,
    gamma=GAMMA,
    epsilon=0,
    epsilon_decay_rate=EPSILON_DECAY_RATE,
    epsilon_min=EPSILON_MIN,
    learning_rate=LEARNING_RATE,
    sample_size=SAMPLE_SIZE,
    batch_size=BATCH_SIZE,
    name=NAME
)

DQNAgent.load()


@app.route('/memory_length', methods=['GET'])
def memory_length():
    memory_length = len(DQNAgent.memory)
    return jsonify(memory_length=memory_length)


@app.route('/update_epsilon', methods=['POST'])
def update_epsilon():
    epsilon = request.get_json()['epsilon']
    DQNAgent.epsilon = epsilon
    return 'ok'


@app.route('/remember', methods=['POST'])
def remember():
    state = np.array(request.get_json()['state'])
    action = request.get_json()['action']
    reward = request.get_json()['reward']
    next_state = np.array(request.get_json()['next_state'])
    done = request.get_json()['done']
    DQNAgent.remember(state, action, reward, next_state, done)
    return 'ok'


@app.route('/replay', methods=['POST'])
def replay():
    if len(DQNAgent.memory) >= SAMPLE_SIZE:
        print("----------------------------------------")
        print("---> Starting experience replay...")
        start_time = time.time()
        losses = []
        accuraces = []
        # DQNAgent.replay()
        for i in range(0, 100):
            loss, acc = DQNAgent.replay()
            losses.append(sum(loss)/len(loss))
            accuraces.append(sum(acc)/len(acc))
            # print(i, '-', 'Loss:', (sum(loss)/len(loss)), 'Accuracy:', (sum(acc)/len(acc)))

        # losses, accuraces = DQNAgent.replay()
        print('--->', 'Loss:', (sum(losses)/len(losses)), 'Accuracy:', (sum(accuraces)/len(accuraces)))

        # print(DQNAgent.epsilon)
        elapsed_time = round(time.time() - start_time, 2)
        print("---> Experience replay took: ", elapsed_time, " seconds")
        # print("----------------------------------------")
    return 'ok'


@app.route('/act', methods=['POST'])
def act():
    states = np.array(request.get_json()['states'])
    action = DQNAgent.act(states)
    if isinstance(action, (np.integer)):
        action = action.item()
    return jsonify(action=action)


@app.route('/save', methods=['POST'])
def save():
    DQNAgent.save()
    return 'ok'


if __name__ == '__main__':
    # Start Web App
    app.run(threaded=False)
