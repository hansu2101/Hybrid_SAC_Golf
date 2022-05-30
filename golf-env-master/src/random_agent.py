import numpy as np
from golf_env import GolfEnv


class RandomAgent:
    def step(self, state):
        return np.random.uniform(-30, 30), np.random.randint(len(GolfEnv.CLUB_INFO))