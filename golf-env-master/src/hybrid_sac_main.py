
from hybrid_sac_learn_2 import SACagent
#from golf_hsac_learn import SACagent
from golf_env_discrete import GolfEnvDiscrete
import tensorflow as tf
def main():

    max_episode_num = 10000
    env = GolfEnvDiscrete()
    agent = SACagent(env)

    with tf.device('/gpu:0'):
        agent.train(max_episode_num)

    agent.plot_result()


if __name__=="__main__":
    main()