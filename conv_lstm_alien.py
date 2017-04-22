# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import sys
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, MaxPooling2D, LSTM, TimeDistributed
from keras import backend as K

# input image dimensions
img_rows, img_cols = 210, 160

if K.image_dim_ordering() == 'th':
    input_shape = (1, 1, img_rows, img_cols)
else:
    input_shape = (1, img_rows, img_cols, 1) 


def printUsage():
    print "Usage: python cartpole.py <num_episodes_to_run>"

if len(sys.argv) < 2:
    print("Not enough arguments: " + str(len(sys.argv)))
    printUsage()
    sys.exit()


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

np.random.seed(7)

#MODEL_DIR_PATH = "./models/"

EPISODES = int(sys.argv[1])

#if (not os.path.exists(MODEL_DIR_PATH)):
#    os.makedirs(MODEL_DIR_PATH)

class DQNAgent:
    def __init__(self, state_size, action_size, state):
        self.state_size = state_size
        self.action_size = action_size
        self.state = state
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        #self.epsilon = 0.0 # adbhat temp
        self.e_decay = .99
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(nb_filter=32, nb_row=5, nb_col=5, subsample=(5,5), border_mode='valid'), input_shape=input_shape))
        model.add(TimeDistributed(Convolution2D(nb_filter=64, nb_row=2, nb_col=2, subsample=(2,2), border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(nb_filter=64, nb_row=2, nb_col=2, subsample=(1,1), border_mode='valid')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(512))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size),True
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]),False  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        X = np.zeros((batch_size, 1, self.state.shape[0], self.state.shape[1], 1))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(next_state)[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, path):
        # self.model.load_weights(name)
        self.model.save(path)

    def save(self, path):
        # self.model.save_weights(path)
        self.model.save(path)


if __name__ == "__main__":
    env = gym.make('Alien-v0')
    observation_space = rgb2gray(env.reset())
    state_size = observation_space.shape[0] * observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, observation_space)

    for episode in xrange(EPISODES):
        
        state = rgb2gray(env.reset())
 
        if K.image_dim_ordering() == 'th':
            state = np.reshape(state, (1, 1, 1, state.shape[0], state.shape[1]))
        else:
            state = np.reshape(state, (1, 1, state.shape[0], state.shape[1], 1))

        sum_reward = 0
        for time in range(1000):
            # env.render()
            action, isRandom = agent.act(state)
            next_state3D, reward, done, _ = env.step(action)
            next_state = rgb2gray(next_state3D)
            if K.image_dim_ordering() == 'th':
                next_state = np.reshape(next_state, (1, 1, 1, next_state.shape[0], next_stat[1]))
            else:
                next_state = np.reshape(next_state, (1, 1, next_state.shape[0], next_state.shape[1], 1))
            
            reward = reward if not done else 0
            sum_reward += reward
            #if time >=199:
                #reward=1
            
            #if episode>700:
             #   total_reward+=reward 

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done or time == 999:
                print("episode:\t{}\ttime:\t{}\te:\t{:.2}\tscore\t{}\tisrandom:\t{}"
                        .format(episode, time, agent.epsilon, sum_reward, isRandom))
                break
                
        agent.replay(32)

    #print("Total:\t{}".format(total_reward))
    #outfileName = "cartpole_v2_l"+str(NUM_HIDDEN_LAYERS)+"_n"+str(NUM_NODES_IN_HIDDEN_LAYER)+"_e"+str(EPISODES)+".h5"
    #agent.save(os.path.join(MODEL_DIR_PATH, outfileName))
