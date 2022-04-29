import gym
#from hybrid_sac_learn import SACagent
from golf_hsac_learn import SACagent
from golf_env_discrete import GolfEnvDiscrete
import tensorflow as tf
import cv2
import numpy as np
import util

def main():

    env = GolfEnvDiscrete()
    agent = SACagent(env)

    agent.load_weights('../save_weights/')

    time = 0
    state = env.reset()

    while True:

        state_img, state_dist = state[0], state[1]
        state_img = np.reshape(cv2.resize(state_img, (84, 84), cv2.INTER_NEAREST), (84, 84)).astype(np.float32)
        state_imgs = np.stack((state_img, state_img, state_img), axis=2)
        state_dist = np.array(state_dist.reshape(1,))

        action_c, action_d = agent.get_action(tf.convert_to_tensor([state_imgs], dtype=tf.float32),
                                              tf.convert_to_tensor([state_dist], dtype=tf.float32))

        print([action_d, action_c])

        next_state, reward, done = env.step((action_c[action_d], action_d), debug=True)
        state = next_state

        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            env.plot()
            break


if __name__=="__main__":
    main()