import sys
import gym
from DqnAgent import DqnAgent
import numpy as np


def preprocess(state):
    """preprocess a state before sending it to the agent"""
    return convert_to_black_and_white(downsample(state))


def downsample(state):
    """
    Take only alternate pixels - basicall halves the resolution of the image
    :param image: 3d numpy array of floats, the input image
    :return 3d numpy array of floats, the downsampled image
    """
    return state[::2, ::2, :]


def convert_to_black_and_white(state):
    """
    convert an rgb state to black and white
    :param state: numpy array, the input image
    return the image converted to black and white
    """
    rcoeff = 0.21
    gcoeff = 0.72
    bcoeff = -.07
    shape = np.array(state.shape)
    shape[2] = 1

    output_state = np.zeros(shape)

    for row in range(shape[0]):
        for col in range(shape[1]):
            output_state[row][col][0] = rcoeff * state[row][col][0] + gcoeff * state[row][col][1] + \
                bcoeff * state[row][col][2]

    return output_state


if __name__ == '__main__':
    env_name = sys.argv[1] if len(sys.argv) > 1 else "Pong-v0"
    env = gym.make(env_name)
    shape = np.array(env.observation_space.shape)
    # we downsample the image and convert it to black and white
    shape[0] /= 2
    shape[1] /= 2
    shape[2] = 1
    shape = tuple(shape)

    agent = DqnAgent(state_size=shape,
                     number_of_actions=env.action_space.n,
                     save_name=env_name)

    num_episodes = 200

    for episode in xrange(num_episodes):

        print("Episode Id:\t{}".format(episode))
        next_state = env.reset()
        next_state = preprocess(next_state)
        #print next_state.shape

        # start a new episode for the agent
        agent.new_episode()

        done = False
        total_cost = 0.0
        total_reward = 0.0
        frame = 0

        while not done:
            frame += 1
            # env.render()

            action = agent.act(next_state)

            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            if reward != 0:
                print "Reward:", reward
            total_cost += agent.observe(reward)
            total_reward += reward

        print "total reward", total_reward
        print "mean cost", total_cost / frame
