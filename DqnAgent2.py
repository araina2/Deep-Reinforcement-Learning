"""
created on Apr 21, 2017 04:20 PM
"""
from collections import deque
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input
from keras.optimizers import RMSprop
import random
from keras.models import load_model
import numpy as np

class DqnAgent:
    """Deep Q-learning Network Agent class"""
    def __init__(self, state_size=None, action_size=1,
                 epsilon=1.0, minibatch_size=32, gamma=0.9, memory_size=50):
        """
        :param state_size: the size of the input states
        :param action_size: int, number of possible actions
        :param epsilon: float, the starting value of epsilon in epsilon greedy approach
        :param minibatch_size: int, size of minibatch on which to perform update
        :param gamma: float, the discount rate
        :param memory_size: int, the amount of previous states to save for replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.e_decay = .99
        self.e_min = 0.05
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        """Neural Net for Deep-Q learning Model"""
        input_layer = Input(shape=self.state_size)
        hlayer1 = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu')(input_layer)
        hlayer2 = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', activation='relu')(hlayer1)
        hlayer3 = Flatten()(hlayer2)
        hlayer4 = Dense(256, activation='relu')(hlayer3)
        output_layer = Dense(self.action_size)(hlayer4)
        model = Model(input_layer, output_layer)

        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        """
        store state, action, rewards etc
        :param state: numpy array, the current state
        :param action: int, index of the action taken
        :param reward: float, reward received for action
        :param next_state: numpy array, the next state
        :param done: boolean, True if episode is complete
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        given an input state, take an action using epsilon greedy method
        :param state: numpy array, the input state
        :return int, the index of the action to take
        """

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), True
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0]), False  # returns action

    def replay(self, batch_size):
        """
        replay memory to train
        :param batch_size: int, the batch size to train on
        """
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size,) + self.state_size)
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(np.array([state]))[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(np.array([next_state]))[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, path):
        """
        load the model
        :param path: string, path to saved model
        """
        self.model = load_model(path)

    def save(self, path):
        """
        save the model
        :param path: string, path to file where the model will be saved
        """
        self.model.save(path)
