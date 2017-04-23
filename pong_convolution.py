"""
created on Apr 21, 2017 04:20 PM
"""
import gym
from DqnAgent2 import DqnAgent
import numpy as np
import time


def convert_to_black_and_white(state):
    """
    convert an rgb state to black and white
    :param state: numpy array, the input image
    return the image converted to black and white
    """
    rcoeff = 0.21
    gcoeff = 0.72
    bcoeff = 0.07
    shape = np.array(state.shape)
    shape[2] = 1

    output_state = np.zeros(shape)
    r, g, b = state[:, :, 0], state[:, :, 1], state[:, :, 2]

    output_state = rcoeff * r + gcoeff * g + bcoeff * b

    return output_state

if __name__ == '__main__':
    env = gym.make('Pong-v0')

    shape = np.array(env.observation_space.shape)
    # we convert the image to black and white
    shape = tuple([shape[0], shape[1], 1])

    agent = DqnAgent(state_size=shape, action_size=env.action_space.n)

    num_episodes = 20000
    save_gap = 10
    batch_size = 256
    save_file = 'pong_conv3_256.h5'

    for episode in xrange(num_episodes):

        print("Episode Id:\t{}".format(episode))
        state = env.reset()
        state = convert_to_black_and_white(state)
        state = np.reshape(state, (1, state.shape[0], state.shape[1], 1))

        done = False
        total_reward = 0
        start_time = time.time()
        while not done:
            #env.render()
            # take action and reward from environment
            action, isRandom = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = convert_to_black_and_white(next_state)
            next_state = np.reshape(state, (1, next_state.shape[0], next_state.shape[1], 1))

            if reward != 0:
                print "Reward:", reward

            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.replay(batch_size)
        end_time = time.time()

        print "Total reward:", total_reward, "Time:", end_time - start_time

        # save the model periodically
        if (episode + 1) % save_gap == 0:
            agent.save(save_file)
