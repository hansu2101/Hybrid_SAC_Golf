from setuptools.command.dist_info import dist_info

from golf_env import GolfEnv
import numpy as np


class HeuristicAgent:

    def __init__(self):
        pass

    def step(self, state):
        dist_to_pin = state[1]
        club_n = len(GolfEnv.CLUB_INFO)

        while True:
            club = np.random.randint(club_n)
            if GolfEnv.CLUB_INFO[club][GolfEnv.ClubInfoIndex.IS_DIST_PROPER](dist_to_pin):
                break

        return np.random.uniform(-45, 45), club