
from hybrid_sac_learn import SACagent
#from golf_hsac_learn import SACagent
from golf_env_discrete import GolfEnvDiscrete

def main():

    max_episode_num = 11000
    env = GolfEnvDiscrete()
    agent = SACagent(env)

    agent.train(max_episode_num)

    agent.plot_result()


if __name__=="__main__":
    main()