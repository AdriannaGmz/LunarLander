import numpy as np
import gym
from PIL import Image

#hyperparameters
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
render = False
alpha = 1
beta = 1

#model initialization
D = 8	# observation space
A = 4	# action space
model = {}
model['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
model['Psi2'] = np.random.randn(A,H) / np.sqrt(H)
model['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
model['Theta2'] = np.random.randn(H) / np.sqrt(H)

#gradbuffer?
#rmsporpchache?


def sigmoid(x): 
	return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

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

def actor_forward(x):
	h = np.dot(model['Psi1'], x)
	h[h<0] = 0
	action_prob = np.dot(model['Psi2'], h)
	output_prob = sigmoid(action_prob)
	return output_prob, h

def actor_backward(action, policy_action, actor_old_probs):
	old_psi2 = model['Psi2']
	new_psi2 = old_psi2 + alpha*(action - policy_action)*actor_old_probs

	old_psi1 = model['Psi1']
	new_psi1 = old_psi1 + alpha*(action - policy_action)*actor_old_probs

	model['Psi1'] = new_psi1
	model['Psi2'] = new_psi2
	return None

def critic_forward(x):
	h = np.dot(model['Theta1'], x)
	h[h<0] = 0
	state_value = np.dot(model['Theta2'], h)
	output_value = sigmoid(state_value)
	return output_value, h

def critic_backward(delta_t, value_old_state):
	old_theta2 = model['Theta2']
	new_theta2 = old_theta2 + beta*delta_t*value_old_state

	old_theta1 = model['Theta1']
	new_theta1 = old_theta1 + beta*delta_t*value_old_state

	model['Theta1'] = new_theta1
	model['Theta2'] = new_theta2
	return None

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

	old_state = observation

	# print(observation)

	actor_probs = actor_forward(old_state)

	policy_action = np.amax(actor_probs)

	action = epsilon_greedy_exploration(policy_action, episode_number)

	observation, reward, done, info = env.step(action)

	new_state = observation

	value_first_state = critic_forward(first_state)
	value_new_state = critic_forward(new_state)

	delta_t = reward + gamma * value_new_state - value_old_state

	critic_backward(delta_t, value_old_state)

	if delta_t > 0:
		actor_backward(action, policy_action, actor_probs)

	if done:
		print('Episode %d finished. Resetting environment...' % episode_number)
		episode_number += 1
		observation = env.reset()



	


	
		
