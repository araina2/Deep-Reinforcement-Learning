import random
import numpy
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K
from theano.gradient import disconnected_grad

class DqnAgent:
    def __init__(self, state_size=None, number_of_actions=1,
                 epsilon=0.1, minibatch_size=32, discount=0.9, memory_size=50,
                 save_name='basic', save_freq=10):
        """
        :param state_size: the size of the input states
        :param number_of_actions: int, number of possible actions
        :param epsilon: float, the starting value of epsilon in epsilon greedy approach
        :param minibatch_size: int, size of minibatch on which to perform update
        :param discount: float, the value of the discount parameter
        :param memory_size: int, the amount of previous states to save for replay
        :param save_name: string
        :param save_freq: int, frequency of saving states
        """
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon  # looks like this value is fixed
        self.minibatch_size = minibatch_size
        self.discount = discount
        self.memory_size = memory_size
        self.save_name = save_name
        # should turn this into a small class itself
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []

        self.episode_id = 1
        self.save_freq = save_freq
        self.build_functions()

    def build_model(self):
        """initialize the neural network structure"""
        input_layer = Input(shape=self.state_size)
        hlayer1 = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu')(input_layer)
        hlayer2 = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='same', activation='relu')(hlayer1)
        hlayer3 = Flatten()(hlayer2)
        hlayer4 = Dense(256, activation='relu')(hlayer3)
        output_layer = Dense(self.number_of_actions)(hlayer4)
        self.model = Model(input_layer, output_layer)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"


    def build_functions(self):
        current_state = Input(shape=self.state_size)
        next_state = Input(shape=self.state_size)
        actions = Input(shape=(1,), dtype='int32')
        rewards = Input(shape=(1,), dtype='float32')
        times = Input(shape=(1,), dtype='int32')

        self.build_model()
        self.value_fn = K.function([current_state], self.model(current_state))

        VS = self.model(current_state)
        VNS = disconnected_grad(self.model(next_state))
        future_value = (1 - times) * VNS.max(axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = rewards + discounted_future_value
        cost = ((VS[:, actions] - target)**2).mean()
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)
        self.train_fn = K.function([current_state, next_state, actions, rewards, times], cost, updates=updates)

    def new_episode(self):
        """start of a new episode, reinitialize all data structures"""
        # TODO: check if we can convert this to numpy
        self.memory_states.append([])
        self.memory_actions.append([])
        self.memory_rewards.append([])
        self.memory_states = self.memory_states[-self.memory_size:]
        self.memory_actions = self.memory_actions[-self.memory_size:]
        self.memory_rewards = self.memory_rewards[-self.memory_size:]
        self.episode_id += 1
        if self.episode_id % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def act(self, state):
        """
        take an action based on the input state
        :param state: numpy array, the input state
        :return the action to take, and the values
        """
        if numpy.random.random() < self.epsilon:
            action = numpy.random.randint(self.number_of_actions)
        else:
            values = self.value_fn([state[None, :]])
            action = values.argmax()

        self.memory_states[-1].append(state)
        self.memory_actions[-1].append(action)
        return action

    def observe(self, reward):
        self.memory_rewards[-1].append(reward)

        # replay train is done after every turn
        return self.replay_train()

    def replay_train(self):
        memory_size = len(self.memory_states)

        batch_states = numpy.zeros((self.minibatch_size,) + self.state_size)
        batch_nextStates = numpy.zeros((self.minibatch_size,) + self.state_size)
        batch_actions = numpy.zeros((self.minibatch_size, 1), dtype=numpy.int32)
        batch_rewards = numpy.zeros((self.minibatch_size, 1), dtype=numpy.float32)
        T = numpy.zeros((self.minibatch_size, 1), dtype=numpy.int32)

        for i in xrange(self.minibatch_size):

            episode = random.randint(max(0, memory_size-50), memory_size-1)

            num_frames = len(self.memory_states[episode])
            frame = random.randint(0, num_frames-1)

            batch_states[i] = self.memory_states[episode][frame]
            T[i] = 1 if frame == num_frames - 1 else 0
            if frame < num_frames - 1:
                batch_nextStates[i] = self.memory_states[episode][frame + 1]
            batch_actions[i] = self.memory_actions[episode][frame]
            batch_rewards[i] = self.memory_rewards[episode][frame]

        cost = self.train_fn([batch_states, batch_nextStates, batch_actions, batch_rewards, T])
        return cost
