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


def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

def noisy_evaluation(theta):
    agent = BinaryActionLinearPolicy(theta)
    rew, T = do_rollout(agent, env, num_steps)
    return rew


# Global variables
agent_action = 0


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    
    env.seed(0)
    np.random.seed(0)
    params = dict(n_iter=10, batch_size=25, elite_frac = 0.2)
    num_steps = 200

    # # directory to write to. For video recording remove env.render in While True and uncomment both lines:
    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)

    agent = MyAgent(env.action_space)



    episode_count = 100
    reward = 0
    done = False

    # for i in range(episode_count):
    #     ob = env.reset()
    #     while True:
    #         action = agent.act(ob, reward, done)
    #         ob, reward, done, _ = env.step(action)
    #         env.render()
    #         if done:
    #             break
    #         # Note there's no env.render() here. But the environment still can open window and
    #         # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    #         # Video is not recorded every episode, see capped_cubic_video_schedule for details.


    # Train the agent, and snapshot each stage
    for (i, iterdata) in enumerate(
        cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
        
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        # if args.display: do_rollout(agent, env, 200, render=True)
        do_rollout(agent, env, 200, render=True)




    # Close the env and write monitor result info to disk
    env.close()
