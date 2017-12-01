#is this now my version? -Dennis

import gym
env = gym.make('LunarLander-v2')
for i_episode in range(20):   			#number of episodes
    observation = env.reset() 			#takes observation from resetting env
    for t in range(100): 				#timesteps per episode
        env.render() 					
        print(observation)
        action = env.action_space.sample() 	#picks in action a random possible action in the action_space
        observation, reward, done, info = env.step(action)  #stores the 4 values that step() can return from the env
# observation (object): an environment-specific object representing 
	# your observation of the environment. 
	# For example, pixel data from a camera, joint angles and joint velocities of a robot, 
	# or the board state in a board game.
# reward (float): amount of reward achieved by the previous action. 
	# The scale varies between environments, but the goal is always to increase 
	# your total reward.
# done (boolean): whether it's time to reset the environment again. Most
	# (but not all) tasks are divided up into well-defined episodes, and
	# done being True indicates the episode has terminated. (For example, perhaps 
	# the pole tipped too far, or you lost your last life.)
# info (dict): diagnostic information useful for debugging. It can sometimes
	# be useful for learning (for example, it might contain the raw probabilities 
	# behind the environment's last state change). However, official evaluations 
	# of your agent are not allowed to use this for learning.

        if done:  							#episode has terminated
            print("Episode finished after {} timesteps".format(t+1))
            break
