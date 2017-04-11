# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop

EPISODES = 5000
timesteps = 100

class DQNAgent:
    def __init__(self, state_size, action_size, state):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.state = state
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = .99
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        X_train3D = np.zeros((1, timesteps, state_size))
        model.add(LSTM(output_dim=20, activation='tanh', 
                      input_shape=X_train3D.shape[1:]))
        model.add(Dense(20, activation='tanh', init='uniform'))
        model.add(Dense(self.action_size, activation='sigmoid'))
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
        X = np.zeros((batch_size, timesteps, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            state3D = np.zeros((batch_size, timesteps, state.shape[1]))
            next_state3D = np.zeros((batch_size, timesteps, state.shape[1]))
            for x in range(state.shape[0]):
                for j in range(timesteps):
                    for k in range(state.shape[1]):
                        if j >= 1:
                            state3D[x][j-1] = state3D[x][j]
                        state3D[x][j][k] = state[x][k]

            
            for x in range(next_state.shape[0]):
                for j in range(timesteps):
                    for k in range(next_state.shape[1]):
                        if j >= 1:
                            next_state3D[x][j-1] = next_state3D[x][j]
                        next_state3D[x][j][k] = next_state[x][k]
            
            target = self.model.predict(state3D)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(next_state3D)[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    #env = gym.make('Pong-v0')
    state = env.observation_space
    state_size = env.observation_space.shape[0] #* env.observation_space.shape[1] * env.observation_space.shape[2]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, state)
    # agent.load("./save/cartpole-master.h5")
    
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state3D = np.zeros((state.shape[0], timesteps, state.shape[1]))
        for i in range(state.shape[0]):
            for time in range(timesteps):
                env.render()
                for k in range(state.shape[1]):
                    if time >= 1:
                        state3D[i][time-1] = state3D[i][time]
                    state3D[i][time][k] = state[i][k]
                
                action = agent.act(state3D)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done or time == 999:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                            .format(e, EPISODES, time, agent.epsilon))
                    break
        agent.replay(32)
        # if e % 10 == 0:
            # agent.save("./save/cartpole.h5") 
