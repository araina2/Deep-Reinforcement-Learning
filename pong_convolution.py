"""
created on Apr 21, 2017 04:20 PM
"""
from pong_v1 import convert_to_black_and_white
import gym
from DqnAgent2 import DqnAgent
import numpy as np
import time


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    shape = np.array(env.observation_space.shape)
    # we convert the image to black and white
    shape[2] = 1
    shape = tuple(shape)

    agent = DqnAgent(state_size=shape, action_size=env.action_space.n)

    num_episodes = 2000
    save_gap = 1
    batch_size = 32
    save_file = 'pong_conv.h5'

    for episode in xrange(num_episodes):
        print("Episode Id:\t{}".format(episode))
        state = env.reset()
        state = convert_to_black_and_white(state)

        done = False
        total_reward = 0
        start_time = time.time()
        while not done:
            # take action and reward from environment
            action, isRandom = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = convert_to_black_and_white(next_state)

            if reward != 0:
                print "Reward:", reward

            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

        end_time = time.time()
        agent.replay(batch_size)
        print "Total reward:", total_reward, "Time:", end_time - start_time

        # save the model periodically
        if (episode + 1) % save_gap == 0:
            agent.save(save_file)
