from golf_env import GolfEnv


class GolfEnvContinuous(GolfEnv):
    def __init__(self):
        super(GolfEnvContinuous, self).__init__()

    def _get_flight_model(self, distance_action):
        return distance_action, 10, 10
