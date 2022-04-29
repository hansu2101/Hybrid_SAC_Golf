import gym
from hybrid_sac_learn import SACagent
import tensorflow as tf

def main():

    env = gym.make("Pendulum-v1")
    agent = SACagent(env)

    agent.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    while True:
        env.render()
        action_c, action_d = agent.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
        action = [action_d, action_c]
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()

