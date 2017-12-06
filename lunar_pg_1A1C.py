""" Trains with Policy gradients. Uses OpenAI Gym. """
# 4 neural networks are used for each action

import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200         # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-4    # learning_rate 
gamma = 0.99    # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False


#models initialization
D = 8 # observation space
A = 4 # action space

modelA = {}
modelA['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA['Psi2'] = np.random.randn(A,H) / np.sqrt(H)

gradA_buffer = { k : np.zeros_like(v) for k,v in modelA.iteritems() } 
rmspropA_cache = { k : np.zeros_like(v) for k,v in modelA.iteritems() } 

 
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def gaussian_sample(mean, std):
  gauss_sample = np.random.normal(mean, std, None)
  return gauss_sample

def epsilon_greedy_exploration(best_action, episode_number):
  epsilon = epsilon_scale/(1 + 0.001 * episode_number)
  prob_vector = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]
  prob_vector[best_action] = epsilon/4 + 1 - epsilon
  action_to_explore = np.random.choice(4, 1, True, prob_vector)
  return action_to_explore

def sample_from_action_probs(action_prob_values):
  cumsum_action = np.cumsum(action_prob_values)
  sum_action = np.sum(action_prob_values)
  #sample_action = np.random.choice(4, 1, True, action_prob_values)
  sample_action = int(np.searchsorted(cumsum_action,np.random.rand(1)*sum_action))
  return sample_action

def take_random_action():
  sampled_action = np.random.randint(4)
  return sampled_action

def discount_rewards(r):      # take 1D float array of rewards and compute discounted reward
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary 
    #   R_t     = r_t + running_add * gamma = r_t + gamma * R_t+1
    running_add = r[t] + running_add * gamma  
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  hA = np.dot(modelA['Psi1'], x)
  hA[hA<0] = 0    # ReLU nonlinearity
  logp = np.dot(modelA['Psi2'], hA)  
  p = sigmoid(logp)
  return p, hA # return probability of taking action 2, and hidden state

def policy_backward(model,eph, epdlogp):
  dPsi2 = np.outer(hA.T, err_probs).T            #(200,4)'   = <(200,1)', (4,)> '
  dhA  = np.dot(err_probs.T, modelA['Psi2']).T   #(1,200)'  = (4,)' (4,200) 
  dhA[hA <= 0] = 0                               # backpro prelu
  dPsi1 = np.dot(np.vstack(dhA), np.vstack(x).T) #(200,8) = < (_1,200_)' (1,8)>
  return {'Psi1':dPsi1, 'Psi2':dPsi2}

env = gym.make("LunarLander-v2")
x = env.reset()
x_prev = None
xs,hAs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0
err_probs = np.zeros(A);

while True:
  step_number += 1

  x_prev = x

  # forward the policy network and sample an action from the returned probabilities
  ac_prob, hA = policy_forward(x)
      # Sample an action from the returned probabilities with greedy exploration
  action = sample_from_action_probs(ac_prob)


  # record various intermediates (needed later for backprop) before stepping
  xs.append(x) 
  hAs.append(hA) 
  for k in range(len(err_probs)):
      err_probs[k] = (1-ac_prob[k]) if k == action else (0-ac_prob[k])  #error in probability of expected label
  dlogps.append(err_probs)
  
  x, reward, done, info = env.step(action) 
  reward_sum += reward
  drs.append(reward) 

  if done: 
    episode_number += 1

    epx     = np.vstack(xs)
    ephA    = np.vstack(hAs)
    epdlogp = np.vstack(dlogps)
    epr     = np.vstack(drs)
    xs,hAs,dlogps,drs = [],[],[],[]

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr 
    grad = policy_backward(modelA, ephA, epdlogp)
    for k in modelA: gradA_buffer[k] += grad[k] # accumulate grad over batch
    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in modelA.iteritems():
        g = gradA_buffer[k] # gradient
        rmspropA_cache[k] = decay_rate * rmspropA_cache[k] + (1 - decay_rate) * g**2
        modelA[k] += alpha * g / (np.sqrt(rmspropA_cache[k]) + 1e-5)
        gradA_buffer[k] = np.zeros_like(v) # reset batch gradient buffer


    # book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished. episode:%d, total steps %d, total reward %f. running mean: %f' % (episode_number-1,step_number-1, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset()
    x_prev = None
    step_number=0
  # print ('ep %d: step %d, rwd %f for action %d' % (episode_number,step_number, reward, action))
