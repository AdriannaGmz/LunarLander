""" Trains an Actor with CACLA learning through a Critic. Uses OpenAI Gym. """
# Two neural networks are used 
# Actor  is NN1, Psi parameters
# Critic is NN2, Theta parameters

import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200         # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99    # discount factor for reward
# decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
# resume = False # resume from previous checkpoint?
render = False

# Initialize Actor (or NN1,A0) and Critic (or NN2,V0) for all states
D = 8*1         # input dimensionality: 8x1 grid
model = {}
model['Psi1']   = np.random.randn(H,D) / np.sqrt(D)
model['Psi2']   = np.random.randn(H) / np.sqrt(H)
model['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
model['Theta2'] = np.random.randn(H) / np.sqrt(H)

xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
epsilon_scale = 1.0

  
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
  h1 = np.dot(model['Psi1'], x)
  h1[h1<0] = 0    # ReLU nonlinearity
  logp = np.dot(model['Psi2'], h1)
  p = sigmoid(logp)
  return p, h1     # return probability of taking action 2, and hidden state

def actor_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dPsi2 = np.dot(eph.T, epdlogp).ravel()
  dh1 = np.outer(epdlogp, model['Psi2'])
  dh1[eph <= 0] = 0 # backpro prelu
  dPsi1 = np.dot(dh1.T, epx)
  return {'Psi1':dPsi1, 'Psi2':dPsi2}

def critic_forward(x,rwd):
  h2 = np.dot(model['Theta1'], x)
  h2[h2<0] = 0      # ReLU nonlinearity
  logp = np.dot(model['Theta2'], h2)
  p = sigmoid(logp)
  return delta_t, h2 # return delta_t and hidden state
  def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
      if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r
      reward_sum += reward
      # record reward (has to be done after we call step() to get reward for previous action)
      drs.append(reward) 

  

def critic_backward(eph, epdlogp):
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['Theta2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'Theta1':dW1, 'Theta2':dW2}



env = gym.make("LunarLander-v2")
observation = env.reset()


while True:
  if render: env.render()

  # ALG. Choose a from policy(s,psi)
  # forward the Actor  and sample an action from the returned probability
  aprob, h1 = actor_forward(observation)
  action    = take_random_action()
  a       = epsilon_greedy_exploration(2, episode_number)
  print('Random Action: %d\nExplore Action: %d' % (action, a))

  # ALG. perform a, observe r and s'
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  delta_t, h2 = critic_forward(x,reward)

  # ALG. calculate delta = r + gamma*V(s') - V(s)
  #   ALG. Theta_t = Theta_t + betta*delta*gradient_V(s)
  #   ALG. if delta > 0 then
  #     ALG. Psi_t = Psi_t + alpha*(a - Ac(s,psi)) grad_Ac(s,psi)
  #   ALG. end if
  #   ALG. If s' is terminal then
  #     ALG   s ~ I 
  #   ALG. else
  #     ALG. s =s'
  #   ALG. end if    






  if done: # an episode finished
    episode_number += 1

