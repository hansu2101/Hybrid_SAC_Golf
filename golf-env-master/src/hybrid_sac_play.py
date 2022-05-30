
from hybrid_sac_learn_3 import SACagent
from golf_env import GolfEnv
import tensorflow as tf
import cv2
import numpy as np
import util

env = GolfEnv()
agent = SACagent(env)
agent.load_weights('../save_weights/')

def main():

    time = 0
    total_reward = 0
    state = env.reset(randomize_initial_pos=True)

    while True:

        state_img, state_dist = state[0], state[1]
        state_img = state_img.astype(np.float32) / 100.0
        state_img = np.stack((state_img, state_img, state_img), axis=2)
        state_dist = np.array(state_dist.reshape(1,))

        mu, _, ac_d = agent.actor(tf.convert_to_tensor([state_img], dtype=tf.float32)
                                  , tf.convert_to_tensor([state_dist], dtype=tf.float32))
        action_c = mu.numpy()[0]
        action_d = np.argmax(ac_d)

        print([action_d, action_c])

        next_state, reward, done = env.step((action_c[action_d], action_d), accurate_shots=True, debug=True)
        state = next_state

        time += 1
        total_reward += reward
        print('Time: ', time, 'Reward: ', reward)

        if done:
            print('total_reward:', total_reward, 'time step:', time)
            env.plot()
            break


if __name__=="__main__":
    for _ in range (1):
        main()