# SAC learn (tf2 subclassing version)
# coded by St.Watermelon

import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, concatenate, Conv2D, MaxPooling2D, Flatten,\
                                    BatchNormalization, Conv2D, Activation, GlobalAveragePooling2D, ZeroPadding2D, Add
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

import cv2

from heuristic_agent import HeuristicAgent
from replaybuffer import ReplayBuffer

##RESNET 50 LAYERS   https://eremo2002.tistory.com/76

class res1_layer(Layer):

    def __init__(self, **kwargs):
        super(res1_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.conv1_pad = ZeroPadding2D(padding=(3, 3))
        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2))
        self.bn_conv1 = BatchNormalization()
        self.activation_1 = Activation('relu')
        self.pool1_pad = ZeroPadding2D(padding=(1, 1))

        super(res1_layer, self).build(input_shape)

    def call(self, inputs):

        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation_1(x)
        x = self.pool1_pad(x)

        return x

class res2_layer(Layer):

    def __init__(self, **kwargs):
        super(res2_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.max_pooling2d_1 = MaxPooling2D((3, 3), 2)
        self.res2a_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2a_branch2a = BatchNormalization()
        self.activation_2 = Activation('relu')

        self.res2a_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2a_branch2b = BatchNormalization()
        self.activation_3 = Activation('relu')

        self.res2a_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.res2a_branch1 = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2a_branch2c = BatchNormalization()
        self.bn2a_branch1 = BatchNormalization()

        self.add_1 = Add()
        self.activation_4 = Activation('relu')

        self.res2b_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2b_branch2a = BatchNormalization()
        self.activation_5 = Activation('relu')

        self.res2b_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2b_branch2b = BatchNormalization()
        self.activation_6 = Activation('relu')

        self.res2b_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2b_branch2c = BatchNormalization()

        self.add_2 = Add()
        self.activation_7 = Activation('relu')

        super(res2_layer, self).build(input_shape)

    def call(self, inputs):

        x = self.max_pooling2d_1(inputs)

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = self.res2a_branch2a(x)
                x = self.bn2a_branch2a(x)
                x = self.activation_2(x)

                x = self.res2a_branch2b(x)
                x = self.bn2a_branch2b(x)
                x = self.activation_3(x)

                x = self.res2a_branch2c(x)
                shortcut = self.res2a_branch1(shortcut)
                x = self.bn2a_branch2c(x)
                shortcut = self.bn2a_branch1(shortcut)

                x = self.add_1([x, shortcut])
                x = self.activation_4(x)

                shortcut = x

            else:
                x = self.res2b_branch2a(x)
                x = self.bn2b_branch2a(x)
                x = self.activation_5(x)

                x = self.res2b_branch2b(x)
                x = self.bn2b_branch2b(x)
                x = self.activation_6(x)

                x = self.res2b_branch2c(x)
                x = self.bn2b_branch2c(x)

                x = self.add_2([x, shortcut])
                x = self.activation_7(x)

                shortcut = x

        return x

class res3_layer(Layer):

    def __init__(self, **kwargs):
        super(res3_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.res3a_branch2a = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')
        self.bn3a_branch2a = BatchNormalization()
        self.activation_11 = Activation('relu')

        self.res3a_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3a_branch2b = BatchNormalization()
        self.activation_12 = Activation('relu')

        self.res3a_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.res3a_branch1 = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')
        self.bn3a_branch2c = BatchNormalization()
        self.bn3a_branch1 = BatchNormalization()

        self.add_4 = Add()
        self.activation_13 = Activation('relu')

        self.res3b_branch2a = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')
        self.bn3b_branch2a = BatchNormalization()
        self.activation_14 = Activation('relu')

        self.res3b_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3b_branch2b = BatchNormalization()
        self.activation_15 =Activation('relu')

        self.res3b_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn3b_branch2c = BatchNormalization()

        self.add_5 = Add()
        self.activation_16 = Activation('relu')

        super(res3_layer, self).build(input_shape)

    def call(self, x):

        shortcut = x

        for i in range(4):
            if (i == 0):
                x = self.res3a_branch2a(x)
                x = self.bn3a_branch2a(x)
                x = self.activation_11(x)

                x = self.res3a_branch2b(x)
                x = self.bn3a_branch2b(x)
                x = self.activation_12(x)

                x = self.res3a_branch2c(x)
                shortcut = self.res3a_branch1(shortcut)
                x = self.bn3a_branch2c(x)
                shortcut = self.bn3a_branch1(shortcut)

                x = self.add_4([x, shortcut])
                x = self.activation_13(x)

                shortcut = x

            else:
                x = self.res3b_branch2a(x)
                x = self.bn3b_branch2a(x)
                x = self.activation_14(x)

                x = self.res3b_branch2b(x)
                x = self.bn3b_branch2b(x)
                x = self.activation_15(x)

                x = self.res3b_branch2c(x)
                x = self.bn3b_branch2c(x)

                x = self.add_5([x, shortcut])
                x = self.activation_16(x)

                shortcut = x

        return x

class res4_layer(Layer):

    def __init__(self, **kwargs):
        super(res4_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.res4a_branch2a = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')
        self.bn4a_branch2a = BatchNormalization()
        self.activation_23 = Activation('relu')

        self.res4a_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4a_branch2b = BatchNormalization()
        self.activation_24 = Activation('relu')

        self.res4a_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.res4a_branch1 = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')
        self.bn4a_branch2c = BatchNormalization()
        self.bn4a_branch1 = BatchNormalization()

        self.add_8 = Add()
        self.activation_25 = Activation('relu')

        self.res4b_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2a = BatchNormalization()
        self.activation_26 = Activation('relu')

        self.res4b_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4b_branch2b = BatchNormalization()
        self.activation_27 =Activation('relu')

        self.res4b_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2c = BatchNormalization()

        self.add_9 = Add()
        self.activation_28 = Activation('relu')

        super(res4_layer, self).build(input_shape)

    def call(self, x):

        shortcut = x

        for i in range(6):
            if (i == 0):
                x = self.res4a_branch2a(x)
                x = self.bn4a_branch2a(x)
                x = self.activation_23(x)

                x = self.res4a_branch2b(x)
                x = self.bn4a_branch2b(x)
                x = self.activation_24(x)

                x = self.res4a_branch2c(x)
                shortcut = self.res4a_branch1(shortcut)
                x = self.bn4a_branch2c(x)
                shortcut = self.bn4a_branch1(shortcut)

                x = self.add_8([x, shortcut])
                x = self.activation_25(x)

                shortcut = x

            else:
                x = self.res4b_branch2a(x)
                x = self.bn4b_branch2a(x)
                x = self.activation_26(x)

                x = self.res4b_branch2b(x)
                x = self.bn4b_branch2b(x)
                x = self.activation_27(x)

                x = self.res4b_branch2c(x)
                x = self.bn4b_branch2c(x)

                x = self.add_9([x, shortcut])
                x = self.activation_28(x)

                shortcut = x

        return x

class res5_layer(Layer):

    def __init__(self, **kwargs):
        super(res5_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.res5a_branch2a = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')
        self.bn5a_branch2a = BatchNormalization()
        self.activation_41 = Activation('relu')

        self.res5a_branch2b = Conv2D(521, (3, 3), strides=(1, 1), padding='same')
        self.bn5a_branch2b = BatchNormalization()
        self.activation_42 = Activation('relu')

        self.res5a_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.res5a_branch1 = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')
        self.bn5a_branch2c = BatchNormalization()
        self.bn5a_branch1 = BatchNormalization()

        self.add_14 = Add()
        self.activation_43 = Activation('relu')

        self.res5b_branch2a = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn5b_branch2a = BatchNormalization()
        self.activation_44 = Activation('relu')

        self.res5b_branch2b = Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.bn5b_branch2b = BatchNormalization()
        self.activation_45 =Activation('relu')

        self.res5b_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.bn5b_branch2c = BatchNormalization()

        self.add_15 = Add()
        self.activation_46 = Activation('relu')

        super(res5_layer, self).build(input_shape)

    def call(self, x):

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = self.res5a_branch2a(x)
                x = self.bn5a_branch2a(x)
                x = self.activation_41(x)

                x = self.res5a_branch2b(x)
                x = self.bn5a_branch2b(x)
                x = self.activation_42(x)

                x = self.res5a_branch2c(x)
                shortcut = self.res5a_branch1(shortcut)
                x = self.bn5a_branch2c(x)
                shortcut = self.bn5a_branch1(shortcut)

                x = self.add_14([x, shortcut])
                x = self.activation_43(x)

                shortcut = x

            else:
                x = self.res5b_branch2a(x)
                x = self.bn5b_branch2a(x)
                x = self.activation_44(x)

                x = self.res5b_branch2b(x)
                x = self.bn5b_branch2b(x)
                x = self.activation_45(x)

                x = self.res5b_branch2c(x)
                x = self.bn5b_branch2c(x)

                x = self.add_15([x, shortcut])
                x = self.activation_46(x)

                shortcut = x

        return x

# actor network
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]  # std bound

        self.res1 = res1_layer()
        self.res2 = res2_layer()
        self.res3 = res3_layer()
        self.res4 = res4_layer()
        self.res5 = res5_layer()

        self.avp = GlobalAveragePooling2D()

        self.x1 = Dense(16, activation='relu')

        self.h1 = Dense(32, activation='relu')
        self.h2 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')
        self.ac_d = Dense(action_dim, activation=None)


    def call(self, state_img, state_dist):


        x = self.res1(state_img)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.avp(x)
        x1 = self.x1(state_dist)

        h = concatenate([x,x1], axis=-1)

        x = self.h1(h)
        x = self.h2(x)

        mu = self.mu(x)
        std = self.std(x)
        ac_d = self.ac_d(x)

        # Scale output to [-action_bound, action_bound]
        mu = Lambda(lambda x: x * self.action_bound)(mu)

        # clipping std
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std, ac_d

    def sample_normal(self, mu, std, ac_d):

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

    def __init__(self,action_dim):
        super(Critic, self).__init__()

        self.res1 = res1_layer()
        self.res2 = res2_layer()
        self.res3 = res3_layer()
        self.res4 = res4_layer()
        self.res5 = res5_layer()

        self.avp = GlobalAveragePooling2D()

        self.x2 = Dense(16, activation='relu')
        self.a1 = Dense(16, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(action_dim, activation='linear')


    def call(self, state_action):
        state_img = state_action[0]
        state_dist = state_action[1]
        action = state_action[2]

        x = self.res1(state_img)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x1 = self.avp(x)
        x2 = self.x2(state_dist)
        a = self.a1(action)

        h = concatenate([x1, x2, a], axis=-1)
        x = self.h2(h)
        x = self.h3(x)
        q = self.q(x)
        return q


class SACagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 128
        self.BUFFER_SIZE = 40000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.01
        self.ALPHA = 0.5
        self.ALPHA_D = 0.5

        self.env = env
        # get state dimension
        # self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = 20
        # get action bound
        self.action_bound = 60

        ## create actor and critic networks
        self.actor = Actor(self.action_dim, self.action_bound)
        # self.actor.build(input_shape=(None, 60, 60, 1))

        self.critic = Critic(self.action_dim)
        self.target_critic = Critic(self.action_dim)

        state_img_in = Input((84, 84, 3))
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
        self.actor.load_weights(path + 'ressactor_1.h5')
        self.critic.load_weights(path + 'resscritic_1.h5')


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.load_weights('../save_weights/')
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            if ep < 7500 :
                state = self.env.reset_randomized(max_timestep=200)

            else :
                state = self.env.reset_limset(max_timestep=100)

            while not done:
                # visualize the environment

                state_img, state_dist = state[0], state[1]
                state_img = np.reshape(cv2.resize(state_img,(84, 84),cv2.INTER_NEAREST), (84, 84)).astype(np.float32)
                state_img = np.stack((state_img, state_img, state_img), axis=2)
                state_dist = np.reshape(state_dist, (1, ))

                if ep > 10000:  # start train after buffer has some amounts
                    action_c, action_d = self.get_action(tf.convert_to_tensor([state_img], dtype=tf.float32),
                                                         tf.convert_to_tensor([state_dist], dtype=tf.float32))
                    # print("actor", action_c[action_d], action_d,  action_c)

                else:
                    action_c, action_d = self.hagent.step(state_dist)
                    # print("heuristic", action_c[action_d], action_d, action_c)



                next_state, reward, done = self.env.step((action_c[action_d], action_d), debug=False)



                next_state_img, next_state_dist = next_state[0], next_state[1]
                next_state_img = np.reshape(cv2.resize(next_state_img,(84, 84),cv2.INTER_NEAREST), (84, 84)).astype(np.float32)
                next_state_img = np.stack((next_state_img, next_state_img, next_state_img), axis=2)
                next_state_dist = np.reshape(next_state_dist, (1,))


                self.buffer.add_buffer(state_img, state_dist, action_d, action_c,
                                       reward, next_state_img, next_state_dist, done)


                if self.buffer.buffer_count() > 130:  # start train after buffer has some amounts

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
            self.actor.save_weights("../save_weights/ressactor_2.h5")
            self.critic.save_weights("../save_weights/resscritic_2.h5")
            # self.env.plot()


        np.savetxt('../save_weights/epi_reward_sac.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()