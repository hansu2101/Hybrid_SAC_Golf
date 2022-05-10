# SAC learn (tf2 subclassing version)
# coded by St.Watermelon

import numpy as np
import matplotlib.pyplot as plt
import torch

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, concatenate, Conv2D, MaxPooling2D, Flatten,\
                                    BatchNormalization, Conv2D, Activation, GlobalAveragePooling2D, ZeroPadding2D, Add
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.initializers import RandomUniform
import torch.distributions as td
import cv2

from heuristic_agent import HeuristicAgent
from replaybuffer import ReplayBuffer
from Resnet50 import res1_layer, res2_layer, res3_layer, res4_layer, res5_layer

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# actor network
class Actor(Model):

    def __init__(self, state_img_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.state_img_dim = state_img_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]  # std bound

        # self.res1 = res1_layer()
        # self.res2 = res2_layer()
        # self.res3 = res3_layer()
        # self.res4 = res4_layer()
        # self.res5 = res5_layer()
        #
        # self.avp = GlobalAveragePooling2D()

        self.model = EfficientNetB0(include_top = False, weights=None, input_shape=self.state_img_dim)
        self.flat = Flatten()
        self.x1 = Dense(400, activation='relu')
        self.x2 = Dense(32, activation='relu')

        self.h1 = Dense(300, activation='relu')
        # self.h2 = Dense(16, activation='relu')

        self.mu = Dense(action_dim, activation='tanh',kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.std = Dense(action_dim, activation='softplus',kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.ac_d = Dense(action_dim, activation=None,kernel_initializer=RandomUniform(-1e-3, 1e-3))


    def call(self, state_img, state_dist):


        # x = self.res1(state_img)
        # x = self.res2(x)
        # x = self.res3(x)
        # x = self.res4(x)
        # x = self.res5(x)
        # x = self.avp(x)
        x1 = self.model(state_img)
        x1 = self.flat(x1)
        x1 = self.x1(x1)
        x2 = self.x2(state_dist)

        h = concatenate([x1, x2], axis=-1)

        x = self.h1(h)
        # x = self.h2(x)

        mu = self.mu(x)
        std = self.std(x)
        ac_d = self.ac_d(x)

        # Scale output to [-action_bound, action_bound]
        mu = Lambda(lambda x: x * self.action_bound)(mu)

        # clipping std
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std, ac_d

    def sample_normal(self, mu, std, ac_d):

        if np.isnan(mu.numpy()[0].all()) or np.isnan(std.numpy()[0].all()) or np.isnan(ac_d.numpy()[0].all()):
            print('IS NAN')
            exit(0)

        normal_prob = tfp.distributions.Normal(mu, std)
        action_c = normal_prob.sample()
        action_c = tf.clip_by_value(action_c, -self.action_bound, self.action_bound)
        log_pdf_c = normal_prob.log_prob(action_c)

        dist = tfp.distributions.Categorical(logits=ac_d)
        action_d = dist.sample()
        prob_d = tf.nn.softmax(dist.logits)
        log_pdf_d = tf.math.log(prob_d + (1e-8))


        return action_c, action_d, log_pdf_c, log_pdf_d, prob_d

# critic network
class Critic(Model):

    def __init__(self,action_dim, state_img_dim):
        super(Critic, self).__init__()

        self.state_img_dim = state_img_dim
        self.model = EfficientNetB0(include_top=False, weights=None, input_shape=self.state_img_dim)
        self.flat = Flatten()
        self.x1 = Dense(400, activation='relu')
        self.x2 = Dense(32, activation='relu')
        self.a1 = Dense(action_dim, activation='relu')
        self.h1 = Dense(300, activation='relu')
        # self.h2 = Dense(16, activation='relu')
        self.q = Dense(action_dim, activation='linear', kernel_initializer=RandomUniform(-1e-3, 1e-3))


    def call(self, state_action):
        state_img = state_action[0]
        state_dist = state_action[1]
        action = state_action[2]

        x1 = self.model(state_img)
        x1 = self.flat(x1)
        x1 = self.x1(x1)
        x2 = self.x2(state_dist)
        a = self.a1(action)

        h = concatenate([x1, x2, a], axis=-1)
        x = self.h1(h)
        # x = self.h3(x)
        q = self.q(x)
        return q


class SACagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.ALPHA = 0.5
        self.ALPHA_D = 0.5

        self.env = env
        # get state dimension
        self.state_img_dim = (80, 80, 3)
        # get action dimension
        self.action_dim = 20
        # get action bound
        self.action_bound_angle = 60

        ## create actor and critic networks
        self.actor = Actor(self.state_img_dim, self.action_dim, self.action_bound_angle)
        # self.actor.build(input_shape=(None, 60, 60, 1))

        self.critic = Critic(self.action_dim, self.state_img_dim)
        self.target_critic = Critic(self.action_dim, self.state_img_dim)

        state_img_in = Input(self.state_img_dim)
        state_dist_in = Input((1,))
        action_in = Input((self.action_dim,))
        self.actor(state_img_in, state_dist_in)
        self.critic([state_img_in,state_dist_in, action_in])
        self.target_critic([state_img_in,state_dist_in, action_in])

        self.actor.summary()
        self.critic.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        ## heuristic_agent
        self.hagent = HeuristicAgent()

        # save the results
        self.save_epi_reward = []



        # target_entropy = -0.25
        # log_alpha = tf.zeros(1, requires_grad=True)
        # alpha = log_alpha.exp().detach()
        # self.a_optimizer = Adam([log_alpha], lr=1e-4)
        #
        # # target_entropy_d = -0.98 * np.log(1/out_d)
        # target_entropy_d = 0.25
        # log_alpha_d = tf.zeros(1, requires_grad=True)
        # alpha_d = log_alpha_d.exp().detach()
        # self.a_d_optimizer = Adam([log_alpha_d], lr=1e-4)

    ## actor policy
    def get_action(self, state_img, state_dist):
        mu, std, ac_d = self.actor(state_img, state_dist)
        action_c, action_d, _ ,_ ,_ = self.actor.sample_normal(mu, std, ac_d)
        return action_c.numpy()[0], action_d.numpy()[0]



    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)


    ## single gradient update on a single batch data
    def critic_learn(self, states_img, states_dist, actions_d, actions_c, q_targets):
        idx = 0
        index = []
        for i in range(len(actions_d)):
            tmp = [idx, int(actions_d.numpy()[i])]
            idx += 1
            index.append(tmp)

        with tf.GradientTape() as tape:
            q = self.critic([states_img, states_dist, actions_c], training=True)
            qgather = tf.gather_nd(q,index)
            loss = tf.reduce_mean(tf.square(qgather-q_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## train the actor network
    def actor_learn(self, states_img, states_dist):
        with tf.GradientTape() as tape:
            mu, std, ac_d = self.actor(states_img, states_dist, training=True)
            actions_c, actions_d, log_pdfs_c, log_pdfs_d, probs_d = self.actor.sample_normal(mu, std, ac_d)
            log_pdfs_c = tf.squeeze(log_pdfs_c)
            log_pdfs_d = tf.squeeze(log_pdfs_d)
            soft_q = self.critic([states_img, states_dist, actions_c])

            loss_c = tf.reduce_mean(probs_d * (self.ALPHA * log_pdfs_c * probs_d - soft_q))
            loss_d = tf.reduce_mean(probs_d * (self.ALPHA_D * log_pdfs_d - soft_q))

            loss = loss_d + loss_c

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## computing soft Q target
    def q_target(self, rewards, q_values, dones):
        q_values_Sum = q_values.sum(1)
        y_k = np.asarray(q_values_Sum)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values_Sum[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.actor.load_weights(path + 'ressactor_3.h5')
        self.critic.load_weights(path + 'resscritic_3.h5')


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            if ep < 4500 :
                state = self.env.reset_randomized(max_timestep=100)

            else :
                state = self.env.reset_limset(max_timestep=100)

            while not done:
                # visualize the environment

                state_img, state_dist = state[0], state[1]
                state_img = np.reshape(cv2.resize(state_img,(80, 80),cv2.INTER_NEAREST), (80, 80)).astype(np.float32) / 100.0
                state_img = np.stack((state_img, state_img, state_img), axis=2)
                state_dist = np.reshape(state_dist, (1, ))

                if ep >= 0:  # start train after buffer has some amounts
                    action_c, action_d = self.get_action(tf.convert_to_tensor([state_img], dtype=tf.float32),
                                                         tf.convert_to_tensor([state_dist], dtype=tf.float32))
                    # print("actor", action_c[action_d], action_d,  action_c)

                else:
                    action_c, action_d = self.hagent.step(state_dist)
                    print("heuristic", action_c[action_d], action_d, action_c)

                if time % 10 == 0 :
                    print('time = ', time , action_d,  action_c[action_d])

                next_state, reward, done = self.env.step((action_c[action_d], action_d), debug=False)



                next_state_img, next_state_dist = next_state[0], next_state[1]
                next_state_img = np.reshape(cv2.resize(next_state_img,(80, 80),cv2.INTER_NEAREST), (80, 80)).astype(np.float32) / 100.0
                next_state_img = np.stack((next_state_img, next_state_img, next_state_img), axis=2)
                next_state_dist = np.reshape(next_state_dist, (1,))


                self.buffer.add_buffer(state_img, state_dist, action_d, action_c,
                                       reward, next_state_img, next_state_dist, done)


                if self.buffer.buffer_count() > 64:  # start train after buffer has some amounts

                    # sample transitions from replay buffer
                    states_img, states_dist, sactions_d, sactions_c, \
                    rewards, next_states_img, next_states_dist, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # predict target soft Q-values
                    next_mu, next_std, next_ac_d = self.actor(tf.convert_to_tensor(next_states_img, dtype=tf.float32),
                                                              tf.convert_to_tensor(next_states_dist, dtype=tf.float32))
                    next_action_c, next_action_d, next_log_pdf_c,next_log_pdf_d, next_prob_d = self.actor.sample_normal(next_mu, next_std, next_ac_d)

                    target_qs = self.target_critic([next_states_img, next_states_dist, next_action_c])
                    target_qi = next_prob_d * (target_qs - self.ALPHA * next_prob_d * next_log_pdf_c - self.ALPHA_D * next_log_pdf_d )

                    # compute TD targets
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    # train critic using sampled batch
                    self.critic_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                      tf.convert_to_tensor(states_dist, dtype=tf.float32),
                                      tf.convert_to_tensor(sactions_d, dtype=tf.float32),
                                      tf.convert_to_tensor(sactions_c, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                    # train actor
                    self.actor_learn(tf.convert_to_tensor(states_img, dtype=tf.float32),
                                     tf.convert_to_tensor(states_dist, dtype=tf.float32))

                    # update both target network
                    self.update_target_network(self.TAU)

                # update current state
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward, 'sample size : ',self.buffer.buffer_count())

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            #print('Now save')
            self.actor.save_weights("../save_weights/ressactor_3.h5")
            self.critic.save_weights("../save_weights/resscritic_3.h5")
            # self.env.plot()


        np.savetxt('../save_weights/epi_reward_sac.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()