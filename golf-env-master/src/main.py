from functools import reduce

import numpy as np

import util
from golf_env_continuous import GolfEnvContinuous
from golf_env_discrete import GolfEnvDiscrete
from heuristic_agent import HeuristicAgent
from random_agent import RandomAgent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = GolfEnvDiscrete()
    agent = HeuristicAgent()

    for _ in range(1):
        state = env.reset()
        # util.show_grayscale(img)

        # ((img, dist), r, term) = env.step((util.deg_to_rad(180), 100), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(30), 100), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(21), 140), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(0), 100), debug=True)
        # util.show_grayscale(img)

        while True:
            state, r, term = env.step(agent.step(state), debug=True)
            # util.show_grayscale(state[0])
            if term:
                break

        env.plot()
