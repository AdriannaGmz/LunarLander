import argparse
import logging
import sys
import numpy as np
from _policies import BinaryActionLinearPolicy # Different file so it can be unpickled

import gym
from gym import wrappers


class MyAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        global agent_action
        agent_action = self.action_space.sample()  # random action to take *by the moment*
        return agent_action           




agent_action = 0


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    # # directory to write to. For video recording remove env.render in While True and uncomment both lines:
    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    
    env.seed(0)
    agent = MyAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
