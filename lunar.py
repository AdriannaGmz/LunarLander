import numpy as np
import gym
from PIL import Image

#hyperparameters
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
render = False

#model initialization
D = 300*200
model = {}
model['A1'] = np.random.randn(H,D) / np.sqrt(D)
model['A2'] = np.random.randn(H) / np.sqrt(H)
model['C1'] = np.random.randn(H,D) / np.sqrt(D)
model['C2'] = np.random.randn(H) / np.sqrt(H)

#gradbuffer?
#rmsporpchache?

def gaussian_sample(mean, std):
	gauss_sample = np.random.normal(mean, std, None)
	return gauss_sample

def epsilon_greedy_exploration(best_action, episode_number):
	epsilon = epsilon_scale/(epsilon_scale + episode_number)
	prob_vector = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]
	prob_vector[best_action] = epsilon/4 + 1 - epsilon
	action_to_explore = np.random.choice(4, 1, True, prob_vector)
 	return action_to_explore

def take_random_action():
	sampled_action = np.random.randint(4)
	return sampled_action


env = gym.make("LunarLander-v2")
observation = env.reset()
frame_minus_1 = None
frame_minus_2 = None
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
frame_number = 0
epsilon_scale = 1.0

while True:
	frame_number+=1
	if render: env.render()

	# print(observation)

	action = take_random_action()
	action2 = epsilon_greedy_exploration(2, episode_number)

	print('Random Action: %d\nExplore Action: %d' % (action, action2))

	episode_number+=1
	if (episode_number == 20):
		break


	


	
		
