
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from keras.applications.efficientnet import EfficientNetB0
from matplotlib import pyplot as plt

from replaybuffer import ReplayBuffer


class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1]

        self.e_net = EfficientNetB0(include_top=False, weights=None)
        self.flat = Flatten()

        self.x1 = Dense(400, activation='relu')
        self.x2 = Dense(20, activation='relu')

        self.h1 = Dense(80, activation='relu')
        self.h2 = Dense(20, activation='relu')
        self.mu = Dense(action_dim, activation='tanh', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.std = Dense(action_dim, activation='softplus', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.ac_d = Dense(action_dim, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))


    def call(self, state_img, state_dist):


        x1 = self.e_net(state_img)
        x1 = self.flat(x1)
        x1 = self.x1(x1)
        x2 = self.x2(state_dist)
        h = concatenate([x1, x2], axis=-1)

        x = self.h1(h)
        x = self.h2(x)
        mu = self.mu(x)
        std = self.std(x)
        ac_d = self.ac_d(x)

        mu = Lambda(lambda x : x * self.action_bound)(mu)

        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std, ac_d

    def sample_normal(self, mu, std, ac_d):
        normal_prob = tfp.distributions.Normal(mu, std)
        action_c = normal_prob.sample()
        action_c = tf.clip_by_value(action_c, -self.action_bound, self.action_bound)
        log_pdf_c = normal_prob.log_prob(action_c)

        dist = tfp.distributions.Categorical(probs=ac_d)
        action_d = dist.sample()
        prob_d = ac_d
        log_pdf_d = tf.math.log(prob_d + (1e-8))

        return action_c, action_d, log_pdf_c, log_pdf_d, prob_d

# critic network
class Critic(Model):

    def __init__(self,action_dim):
        super(Critic, self).__init__()

        self.e_net = EfficientNetB0(include_top=False, weights=None)
        self.flat = Flatten()
        self.x1 = Dense(400, activation='relu')
        self.x2 = Dense(20, activation='relu')
        self.a1 = Dense(20, activation='relu')

        self.h1 = Dense(80, activation='relu')
        self.h2 = Dense(20, activation='relu')
        self.q = Dense(action_dim, activation='linear')


    def call(self, state_action):
        state_img = state_action[0]
        state_dist = state_action[1]
        action = state_action[2]

        x1 = self.e_net(state_img)
        x1 = self.flat(x1)
        # x1 = self.x1(x1)
        x2 = self.x2(state_dist)
        a = self.a1(action)

        h = concatenate([x1, x2, a], axis=-1)
        x = self.h1(h)
        x = self.h2(x)
        q = self.q(x)

        return q


class SACagent(object):

    def __init__(self, env):

        self.GAMMA = 0.95
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 30000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.0001
        self.ALPHA = 0.5
        self.ALPHA_D = 0.5

        self.env = env
        self.state_dim_img = (84, 84, 3)
        self.state_dim_dist = (1,)
        self.action_dim = 20
        self.action_bound = 35

        self.episodes = []
        self.actor_loss = []
        self.critic1_loss = []
        self.critic2_loss = []
        self.actor_loss_avg = []
        self.critic1_loss_avg = []
        self.critic2_loss_avg = []
        self.ep_temp = 0

        self.actor = Actor(self.action_dim, self.action_bound)

        self.critic1 = Critic(self.action_dim)
        self.target_critic1 = Critic(self.action_dim)
        self.critic2 = Critic(self.action_dim)
        self.target_critic2 = Critic(self.action_dim)

        state_in_img = Input(self.state_dim_img)
        state_in_dist = Input(self.state_dim_dist)
        action_in = Input(self.action_dim)

        self.actor(state_in_img, state_in_dist)
        self.critic1([state_in_img, state_in_dist, action_in])
        self.target_critic1([state_in_img, state_in_dist, action_in])
        self.critic2([state_in_img, state_in_dist, action_in])
        self.target_critic2([state_in_img, state_in_dist, action_in])

        self.actor.summary()
        self.critic1.summary()
        self.critic2.summary()

        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt1 = Adam(self.CRITIC_LEARNING_RATE)
        self.critic_opt2 = Adam(self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        #self.hagent = HeuristicAgent()

        self.save_epi_reward = []

    def get_action(self, state_img, state_dist):
        mu, std, ac_d = self.actor(state_img, state_dist)
        action_c, action_d, _ ,_ ,_ = self.actor.sample_normal(mu, std, ac_d)
        return action_c.numpy()[0], action_d.numpy()[0]


    def update_target_network(self, TAU):
        phi1 = self.critic1.get_weights()
        phi2 = self.critic2.get_weights()
        target_phi1 = self.target_critic1.get_weights()
        target_phi2 = self.target_critic2.get_weights()

        for i in range(len(phi1)):
            target_phi1[i] = TAU * phi1[i] + (1 - TAU) * target_phi1[i]
            target_phi2[i] = TAU * phi2[i] + (1 - TAU) * target_phi2[i]

        self.target_critic1.set_weights(target_phi1)
        self.target_critic2.set_weights(target_phi2)

    def critic_learn(self, states_img, states_dist, actions_d, actions_c, q_targets):
        idx = 0
        index = []
        for i in range(len(actions_d)):
            tmp = [idx, int(actions_d.numpy()[i])]
            idx += 1
            index.append(tmp)

        with tf.GradientTape() as tape:
            q1 = self.critic1([states_img, states_dist, actions_c], training=True)
            qgather1 = tf.gather_nd(q1, index)
            loss1 = tf.reduce_mean(tf.square(qgather1-q_targets))
        self.critic1_loss.append(loss1)

        grads1 = tape.gradient(loss1, self.critic1.trainable_variables)
        self.critic_opt1.apply_gradients(zip(grads1, self.critic1.trainable_variables))

        with tf.GradientTape() as tape:
            q2 = self.critic2([states_img, states_dist, actions_c], training=True)
            qgather2 = tf.gather_nd(q2, index)
            loss2 = tf.reduce_mean(tf.square(qgather2-q_targets))
        self.critic2_loss.append(loss2)

        grads2 = tape.gradient(loss2, self.critic2.trainable_variables)
        self.critic_opt2.apply_gradients(zip(grads2, self.critic2.trainable_variables))

    def actor_learn(self, states_img, states_dist):
        with tf.GradientTape() as tape:
            mu, std, ac_d = self.actor(states_img, states_dist, training=True)
            actions_c, actions_d, log_pdfs_c, log_pdfs_d, probs_d = self.actor.sample_normal(mu, std, ac_d)
            log_pdfs_c = tf.squeeze(log_pdfs_c)
            log_pdfs_d = tf.squeeze(log_pdfs_d)

            soft_q1 = self.critic1([states_img, states_dist, actions_c])
            soft_q2 = self.critic2([states_img, states_dist, actions_c])
            soft_q = tf.math.minimum(soft_q1, soft_q2)

            loss_c = tf.reduce_mean(probs_d * (self.ALPHA * log_pdfs_c * probs_d - soft_q))
            loss_d = tf.reduce_mean(probs_d * (self.ALPHA_D * log_pdfs_d - soft_q))

            loss = loss_d + loss_c
        self.actor_loss.append(loss)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))



    def q_target(self, rewards, q_values, dones):

        q_values_Sum = q_values.sum(1)
        y_k = np.asarray(q_values_Sum)

        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values_Sum[i]
        return y_k


    def load_weights(self, path):

        self.actor.load_weights(path + 'actor_e_net_a.h5')
        self.critic1.load_weights(path + 'critic_e_net_q1.h5')
        self.critic2.load_weights(path + 'critic_e_net_q2.h5')


    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.load_weights('../save_weights/')
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False

            if ep < 100000:
                if ep % 100==0:
                    state = self.env.reset(randomize_initial_pos=False, max_timestep=30, animation_path='../animation/'+'s_'+str(ep)+'.gif')
                else:
                    state = self.env.reset(randomize_initial_pos=False, max_timestep=35)

            else:
                state = self.env.reset()

            while not done:

                state_img, state_dist = state[0], state[1]
                state_img = state_img.astype(np.float32)/100.0
                state_img = np.stack((state_img, state_img, state_img), axis=2)
                state_dist = np.reshape(state_dist, (1, ))

                action_c, action_d = self.get_action(tf.convert_to_tensor([state_img], dtype=tf.float32),
                                                         tf.convert_to_tensor([state_dist], dtype=tf.float32))

                next_state, reward, done = self.env.step((action_c[action_d], action_d), regenerate_heuristic_club_availability=False, debug=True)

                next_state_img, next_state_dist = next_state[0], next_state[1]
                next_state_img = next_state_img.astype(np.float32) / 100.0
                next_state_img = np.stack((next_state_img, next_state_img, next_state_img), axis=2)
                next_state_dist = np.reshape(next_state_dist, (1,))

                self.buffer.add_buffer(state_img, state_dist, action_d, action_c,
                                        reward, next_state_img, next_state_dist, done)

                if self.buffer.buffer_count() > 500:
                    if self.buffer.buffer_count() == 501:
                        self.ep_temp = ep

                    states_img, states_dist, sactions_d, sactions_c, rewards, next_states_img, next_states_dist, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    next_mu, next_std, next_ac_d = self.actor(tf.convert_to_tensor(next_states_img, dtype=tf.float32),
                                                              tf.convert_to_tensor(next_states_dist, dtype=tf.float32))
                    next_action_c, next_action_d, next_log_pdf_c,next_log_pdf_d, next_prob_d = self.actor.sample_normal(next_mu, next_std, next_ac_d)

                    target_qs1 = self.target_critic1([next_states_img, next_states_dist, next_action_c])
                    target_qs2 = self.target_critic2([next_states_img, next_states_dist, next_action_c])
                    target_qs = tf.math.minimum(target_qs1, target_qs2)

                    target_qi = next_prob_d * (target_qs - self.ALPHA * next_prob_d * next_log_pdf_c - self.ALPHA_D * next_log_pdf_d )

                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    self.critic_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                      tf.convert_to_tensor(states_dist, dtype=tf.float32),
                                      tf.convert_to_tensor(sactions_d, dtype=tf.float32),
                                      tf.convert_to_tensor(sactions_c, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    self.actor_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                     tf.convert_to_tensor(states_dist, dtype=tf.float32))

                    self.update_target_network(self.TAU)

                episode_reward += reward
                state = next_state
                time += 1

            # if self.buffer.buffer_count() > 5000:
            #
            #     self.actor_loss_avg.append(np.mean(self.actor_loss))
            #     self.actor_loss.clear()
            #     self.critic1_loss_avg.append(np.mean(self.critic1_loss))
            #     self.critic1_loss.clear()
            #     self.critic2_loss_avg.append(np.mean(self.critic2_loss))
            #     self.critic2_loss.clear()
            #     self.episodes.append(ep)
            #
            #     if ep % 400 == 0:
            #         plt.figure(figsize=(10, 10))
            #         plt.xlabel('Episode')
            #         plt.ylabel('Loss')
            #         plt.xlim([self.ep_temp, len(self.episodes) + self.ep_temp])
            #         plt.plot(self.episodes, self.critic1_loss_avg, color="red", label='Critic1_Loss')
            #         plt.plot(self.episodes, self.critic2_loss_avg, color="blue", label='Critic2_Loss')
            #         plt.legend()
            #         plt.savefig('../graph/Critic_Loss.png')

                    # plt.figure(figsize=(10, 10))
                    # plt.xlabel('Episode')
                    # plt.ylabel('Loss')
                    # plt.xlim([self.ep_temp, len(self.episodes) + self.ep_temp])
                    # plt.plot(self.episodes, self.actor_loss_avg, color="black", label='Actor_Loss')
                    # plt.legend()
                    # plt.savefig('../graph/Actor_Loss.png')

            print('Episode:', ep+1, 'Time:', time, 'Reward:', episode_reward, 'sample size: ', self.buffer.buffer_count())

            if ep % 10 == 0:
                self.actor.save_weights("../save_weights/actor_e_net_a.h5")
                self.critic1.save_weights("../save_weights/critic_e_net_q1.h5")
                self.critic2.save_weights("../save_weights/critic_e_net_q2.h5")