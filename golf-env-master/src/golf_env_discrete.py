from golf_env import GolfEnv


class GolfEnvDiscrete(GolfEnv):
    def __init__(self):
        super(GolfEnvDiscrete, self).__init__()
        self.flight_models = (
            # NAME    DIST    DEV_X   DEV_Y         exp((x-50)/95)+2 --> 250 10m , 50 3m
            ('DR',    230,    60/3,     8.6/3),
            ('W3',    215,    55/3,     7.6/3),
            ('W5',    195,    40/3,     6.6/3),
            ('I3',    180,    40/3,     5.9/3),
            ('I4',    170,    35/3,     5.5/3),
            ('I5',    160,    30/3,     5.1/3),
            ('I6',    150,    30/3,     4.8/3),
            ('I7',    140,    30/3,     4.5/3),
            ('I8',    130,    30/3,     4.3/3),
            ('I9',    115,    35/3,     3.9/3),
            ('PW',    105,    40/3,     3.7/3),
            ('SW9',    80,     40/3,     3.3/3),
            ('SW8',    70,     35/3,     3.2/3),
            ('SW7',    60,     30/3,     3.1/3),
            ('SW6',    50,     20/3,     3.0/3),
            ('SW5',    40,     15/3,     2.9/3),
            ('SW4',    30,     10/3,     2.8/3),
            ('SW3',    20,     5/3,      2.7/3),
            ('SW2',    10,     3/3,      2.65/3),
            ('SW1',    5,      1/3,      2.6/3),
        )
        self.selected_club_info = None

    def _get_flight_model(self, distance_action):
        self.selected_club_info = self.flight_models[distance_action]
        return self.selected_club_info[1:4]  # exclude name

    def _generate_debug_str(self, msg):
        return 'used' + str(self.selected_club_info) + ' ' + msg
