# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

np.random.seed(5)

EPISODES = 1000
NUM_HIDDEN_LAYERS = sys.argv[1] #has to be atleast 1
NUM_NODES_IN_HIDDEN_LAYER = sys.argv[2]
EPISODES = int(sys.argv[3])

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
        model.add(Dense(int(NUM_NODES_IN_HIDDEN_LAYER), input_dim=self.state_size, activation='tanh'))
        for i in range(int(NUM_HIDDEN_LAYERS)-1):
            model.add(Dense(int(NUM_NODES_IN_HIDDEN_LAYER), activation='tanh', init='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    print 'NUM_NODES: ',str(NUM_NODES_IN_HIDDEN_LAYER)
    print 'NUM_HIDDEN_LAYERS: ',str(NUM_HIDDEN_LAYERS)
    env = gym.make('CartPole-v0')
    #env = gym.make('Pong-v0')
    state_size = env.observation_space.shape[0] #* env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-master.h5")

    for e in range(EPISODES):
        state = env.reset()
        #print 'Init state',state
        state = np.reshape(state, [1, state_size])
        for time in range(1000):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == 999:
                print("episode:\t{}\tscore:\t{}\te:\t{:.2}"
                        .format(e, time, agent.epsilon))
                break
        agent.replay(32)
        # if e % 10 == 0:
            # agent.save("./save/cartpole.h5")
