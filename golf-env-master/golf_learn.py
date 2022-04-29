# DDPG Actor (tf2 subclassing version: using chain rule to train Actor)
# coded by St.Watermelon

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate , Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from golf_env.src.random_agent import RandomAgent

from replaybuffer import ReplayBuffer


# actor network
class Actor(Model):

    def __init__(self, action_dim, action_bound_angle, action_bound_dist):
        super(Actor, self).__init__()

        self.action_bound_angle = action_bound_angle
        self.action_bound_dist = action_bound_dist

        self.c1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu',padding='same')
        self.c2 = MaxPooling2D((2,2))
        self.c3 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu',padding='same')
        self.c4 = MaxPooling2D((2,2))
        self.c5 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',padding='same')
        self.c6 = MaxPooling2D()
        self.c7 = Flatten()

        self.x1 = Dense(128, activation='relu')
        self.b1 = BatchNormalization()
        self.x2 = Dense(128, activation='relu')
        self.b2 = BatchNormalization()

        self.h1 = Dense(64, activation='relu')
        self.b3 = BatchNormalization()
        self.h2 = Dense(32, activation='relu')
        self.b4 = BatchNormalization()
        self.h3 = Dense(16, activation='relu', bias_initializer=
                        tf.keras.initializers.random_uniform(
                            minval=-0.003, maxval=0.003))
        self.b5 = BatchNormalization()

        self.action = Dense(2, activation='tanh')


    def call(self, state_img, state_dist):

        x = self.c1(state_img)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)

        x1 = self.x1(x)
        x1 = self.b1(x1)

        x2 = self.x2(state_dist)
        x2 = self.b2(x2)

        h = concatenate([x1, x2], axis=-1)

        x = self.h1(h)
        x = self.b3(x)
        x = self.h2(x)
        x = self.b4(x)
        x = self.h3(x)
        x = self.b5(x)

        action = self.action(x)

        print(action)

        return action


# critic network
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.c1 = Conv2D(32, (8, 8), strides=(4, 4),padding='same', activation='relu')
        self.c2 = MaxPooling2D((2,2))
        self.c3 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu',padding='same')
        self.c4 = MaxPooling2D((2,2))
        self.c5 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',padding='same')
        self.c6 = MaxPooling2D()
        self.c7 = Flatten()

        self.x1 = Dense(128, activation='relu')
        self.b1 = BatchNormalization()

        self.x2 = Dense(128, activation='relu')
        self.b2 = BatchNormalization()

        self.a1 = Dense(128, activation='relu')
        self.b3 = BatchNormalization()

        self.h2 = Dense(64, activation='relu')
        self.b4 = BatchNormalization()
        self.h3 = Dense(32, activation='relu',  bias_initializer=
                        tf.keras.initializers.random_uniform(
                            minval=-0.003, maxval=0.003))
        self.b5 = BatchNormalization()
        self.q = Dense(1, activation='linear')

    def call(self, state_img, state_dist, action):
        state_img = state_img
        state_dist = state_dist
        action = action

        x = self.c1(state_img)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)

        x1 = self.x1(x)
        x1 = self.b1(x1)

        x2 = self.x2(state_dist)
        x2 = self.b2(x2)

        a = self.a1(action)
        a = self.b3(a)

        h = concatenate([x1, x2, a], axis=-1)

        x = self.h2(h)
        x = self.b4(x)
        x = self.h3(x)
        x = self.b5(x)
        q = self.q(x)

        print(q)
        return q


class DDPGagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 50000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.env = env
        # get state dimension
        self.state_dim_img = (300, 300, 1)
        self.state_dim_dist = (1,)
        # get action dimension
        self.action_dim = (2,)
        # get action bound
        self.action_bound_angle = 90
        self.action_bound_dist = 200

        # create actor and critic networks
        self.actor = Actor(self.action_dim, self.action_bound_angle, self.action_bound_dist)
        self.target_actor = Actor(self.action_dim, self.action_bound_angle, self.action_bound_dist)

        self.critic = Critic()
        self.target_critic = Critic()

        state_in_as1 = Input(self.state_dim_img)
        state_in_as2 = Input(self.state_dim_dist)
        self.actor(state_in_as1, state_in_as2)
        self.target_actor(state_in_as1, state_in_as2)

        state_in = Input(self.state_dim_img)
        state_in2 = Input(self.state_dim_dist)
        action_in = Input(self.action_dim)
        self.critic(state_in, state_in2, action_in)
        self.target_critic(state_in, state_in2, action_in)

        self.actor.summary()
        self.critic.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.RDagent = RandomAgent()

        # save the results
        self.save_epi_reward = []


    def update_target_network(self, TAU):
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        self.target_actor.set_weights(target_theta)

        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)


    def critic_learn(self, states_img, states_dist, actions, td_targets):        #DQN과 유사하고 Loss를 최소화 시키는 방향
        with tf.GradientTape() as tape:
            q = self.critic(states_img, states_dist, actions, training=True)
            loss = tf.reduce_mean(tf.square(q-td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def actor_learn(self, states_img, states_dist):
        with tf.GradientTape() as tape:
            actions = self.actor(states_img, states_dist, training=True)
            critic_q = self.critic(states_img, states_dist, actions)
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)

    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    def load_weights(self, path):
        self.actor.load_weights(path + 'golf_actor.h5')
        self.critic.load_weights(path + 'golf_critic.h5')


    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.load_weights('./save_weights/')
        self.update_target_network(1.0)


        for ep in range(int(max_episode_num)):
            # reset OU noise
            noise, pre_noise = np.zeros(self.action_dim)
            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset()
            state_img, state_dist = state[0], state[1]
            state_img = np.array(state_img.reshape(self.state_dim_img))
            state_dist = np.array(state_dist.reshape(self.state_dim_dist))

            while not done:

                if ep >= 999:

                    state_img, state_dist = state[0], state[1]
                    state_img = np.array(state_img.reshape(self.state_dim_img))
                    state_dist = np.array(state_dist.reshape(self.state_dim_dist))

                    action = self.actor(tf.convert_to_tensor([state_img], dtype=tf.float32),
                                        tf.convert_to_tensor([state_dist], dtype=tf.float32))
                    action = action.numpy()[0]  # 텐서형태를 넘파이로
                    noise = self.ou_noise(pre_noise, dim=self.action_dim)

                    action = action + noise.flatten()
                    action = np.clip(action, a_min=[-1, -1], a_max=[1, 1])

                    # observe reward, new_state
                    next_state, reward, done = self.env.step(action, debug=True)

                    next_state_img, next_state_dist = next_state[0], next_state[1]
                    next_state_img = np.array(next_state_img.reshape(self.state_dim_img))
                    next_state_dist = np.array(next_state_dist.reshape(self.state_dim_dist))

                    # add transition to replay buffer
                    self.buffer.add_buffer(state_img, state_dist, action, reward, next_state_img, next_state_dist, done)

                    if self.buffer.buffer_count() > 2500:

                        # sample transitions from replay buffer
                        states_img, states_dist, actions, rewards, next_states_img, next_states_dist, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                        # predict target Q-values
                        target_qs = self.target_critic(tf.convert_to_tensor(next_states_img, dtype=tf.float32),
                                                       tf.convert_to_tensor(next_states_dist, dtype=tf.float32),
                                                        self.target_actor(
                                                            tf.convert_to_tensor(next_states_img, dtype=tf.float32),
                                                            tf.convert_to_tensor(next_states_dist, dtype=tf.float32)))
                        # compute TD targets
                        y_i = self.td_target(rewards, target_qs.numpy(), dones)

                        # train critic using sampled batch
                        self.critic_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                          tf.convert_to_tensor(states_dist, dtype=tf.float32),
                                          tf.convert_to_tensor(actions, dtype=tf.float32),
                                          tf.convert_to_tensor(y_i, dtype=tf.float32))
                        # train actor
                        self.actor_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                         tf.convert_to_tensor(states_dist, dtype=tf.float32))
                        # update both target network
                        self.update_target_network(self.TAU)
                    # update current state
                    pre_noise = noise

                    state_img = next_state_img
                    state_dist = next_state_dist

                    episode_reward += reward
                    time += 1

                else:

                    angle, dist = self.RDagent.step()
                    action = np.array([angle, dist]).reshape(2, )

                    ((next_state_img, next_state_dist), reward, done, area) = self.env.step(action, debug=True)
                    next_state_img = np.array(next_state_img.reshape(self.state_dim_img))
                    next_state_dist = np.array(next_state_dist.reshape(self.state_dim_dist))

                    self.buffer.add_buffer(state_img, state_dist, action, reward, next_state_img, next_state_dist, done)

                    if self.buffer.buffer_count() > 2500:

                        # sample transitions from replay buffer
                        states_img, states_dist, actions, rewards, next_states_img, next_states_dist, dones = self.buffer.sample_batch(self.BATCH_SIZE)


                        # predict target Q-values
                        target_qs = self.target_critic(tf.convert_to_tensor(next_states_img, dtype=tf.float32),
                                                       tf.convert_to_tensor(next_states_dist, dtype=tf.float32),
                                                        self.target_actor(
                                                            tf.convert_to_tensor(next_states_img, dtype=tf.float32),
                                                            tf.convert_to_tensor(next_states_dist, dtype=tf.float32)))
                        # compute TD targets
                        y_i = self.td_target(rewards, target_qs.numpy(), dones)

                        # train critic using sampled batch
                        self.critic_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                          tf.convert_to_tensor(states_dist, dtype=tf.float32),
                                          tf.convert_to_tensor(actions, dtype=tf.float32),
                                          tf.convert_to_tensor(y_i, dtype=tf.float32))
                        # train actor
                        self.actor_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                         tf.convert_to_tensor(states_dist, dtype=tf.float32))
                        # update both target network
                        self.update_target_network(self.TAU)

                    state_img = next_state_img
                    state_dist = next_state_dist

                    episode_reward += reward
                    time += 1
                    # rewards += r
                    # print(ep, "'s  rewards ", rewards)
                    # env.plot()



            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)
            if ep % 5 == 0:
                #print('Now save')
                self.actor.save_weights("./save_weights/golf_actor.h5")
                self.critic.save_weights("./save_weights/golf_critic.h5")

        np.savetxt('./save_weights/golf_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()