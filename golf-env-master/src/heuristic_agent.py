import numpy as np
import random


class HeuristicAgent:

    def __init__(self):
        pass

    def step(self, state):
        distance = state

        # if distance > 300:
        #     club = np.random.randint(0, 5)
        # elif distance > 200:
        #     club = np.random.randint(5, 7)
        # elif distance > 100:
        #     club = np.random.randint(1, 20)
        # elif distance > 70:
        #     club = np.random.randint(10, 20)
        # else:
        #     club = np.random.randint(11, 20)

        club = np.random.randint(0, 20)

        temp = random.random()
        angle_temp = -35 + 70 * temp
        angle = np.ones(20) * angle_temp
        # np.random.uniform(-35, 35, size=20)
        return angle, club
