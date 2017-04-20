# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import sys
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

def printUsage():
    print "Usage: python cartpole.py <num_hidden_layers> <num_nodes_in_hidden_layer> <num_episodes_to_run>"

if len(sys.argv) < 4:
    print("Not enough arguments: " + str(len(sys.argv)))
    printUsage()
    sys.exit()

np.random.seed(7)

MODEL_DIR_PATH = "./models/"

NUM_HIDDEN_LAYERS = int(sys.argv[1]) # has to be atleast 1
NUM_NODES_IN_HIDDEN_LAYER = int(sys.argv[2])
EPISODES = int(sys.argv[3])

if (not os.path.exists(MODEL_DIR_PATH)):
    os.makedirs(MODEL_DIR_PATH)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
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
        model.add(Dense(NUM_NODES_IN_HIDDEN_LAYER, input_dim=self.state_size, activation='tanh'))
        for i in range(NUM_HIDDEN_LAYERS-1):
            model.add(Dense(NUM_NODES_IN_HIDDEN_LAYER, activation='tanh', kernel_initializer='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
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
        X = np.zeros((batch_size, self.state_size))
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
    # print 'NUM_NODES: ',str(NUM_NODES_IN_HIDDEN_LAYER)
    # print 'NUM_HIDDEN_LAYERS: ',str(NUM_HIDDEN_LAYERS)
    env = gym.make('CartPole-v0')
    
    state_size = env.observation_space.shape[0] #* env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    total_reward=0

    for episode in xrange(EPISODES):
        
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(1000):
            # env.render()
            action, isRandom = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            reward = reward if not done else -10
            if time >=199:
                reward=1
            
            if episode>700:
                total_reward+=reward 

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done or time == 999:
                print("episode:\t{}\ttime:\t{}\te:\t{:.2}\ttotal_reward\t{}"
                        .format(episode, time, agent.epsilon, total_reward))
                break
                
        agent.replay(32)

    print("Total:\t{}".format(total_reward))
    outfileName = "cartpole_v2_l"+str(NUM_HIDDEN_LAYERS)+"_n"+str(NUM_NODES_IN_HIDDEN_LAYER)+"_e"+str(EPISODES)+".h5"
    agent.save(os.path.join(MODEL_DIR_PATH, outfileName))
