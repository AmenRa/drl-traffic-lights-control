#!/usr/bin/python3
import time
import numpy as np
import random
import requests
from ast import literal_eval
from functools import reduce
import math
import random

next_states = np.random.rand(3, 2, 2)

print(next_states)

next_states = np.delete(next_states, 0, 0)

print(next_states)
