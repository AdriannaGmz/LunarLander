import numpy as np
import gym

#hyperparameters
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
render = False

#model initialization
D = 600*400
model = {}
model['A1'] = np.random.randn(H,D) / np.sqrt(D)
model['A2'] = np.random.randn(H) / np.sqrt(H)
model['C1'] = np.random.randn(H,D) / np.sqrt(D)
model['C2'] = np.random.randn(H) / np.sqrt(H)

#gradbuffer?
#rmsporpchache?

def prepro(I):
	#no cropping
	#background is alrady black
	I[I != 0] = 1 #make other stuff white
	return I.astype(np.float).ravel()

def gaussian_sample(mean, std):
	gauss_sample = np.random.normal(mean, std, None)
	return gauss_sample 

env = gym.make("Pong-v0")
observation = env.reset()
frame_minus_1 = None
frame_minus_2 = None
frame_minus_3 = None
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

while True:
	if render: env.render()

	# preprocess observaiton
	frame_current = prepro(observation)

	if frame_minus_2 is None
	
		if frame_minus_1 is None
			frame_diff_vel_2 = np.zeros(D) - np.zeros(D)
		else
			frame_diff_vel_2 = frame_minus_1 - np.zeros(D)
	
	else
		frame_diff_vel_2 = frame_minus_1 - frame_minus_2

	frame_diff_acc = frame_diff_vel_1 - frame_diff_vel_2

	frame_minus_2 = frame_minus_1
	frame_minus_1 = frame_current
	
		
