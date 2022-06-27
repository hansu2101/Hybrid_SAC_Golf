from hybrid_sac_learn_3 import SACagent
from golf_env.src.golf_env import GolfEnv
import tensorflow as tf

def main():
    max_episode_num = 100000
    env = GolfEnv('sejong')
    agent = SACagent(env)

    with tf.device('/gpu:0'):
        agent.train(max_episode_num)

    agent.plot_result()


if __name__=="__main__":
    main()