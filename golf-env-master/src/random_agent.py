import numpy as np


class RandomAgent:
    def step(self):
        return np.random.uniform(-90, 90), np.random.uniform(30, 300)
