#!/usr/bin/python3
import time
import numpy as np
import random
import requests
from ast import literal_eval
from functools import reduce
import math


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


print(sigmoid(0.123))
