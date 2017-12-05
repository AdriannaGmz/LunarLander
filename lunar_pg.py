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

modelA0 = {}
modelA0['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA0['Psi2'] = np.random.randn(H) / np.sqrt(H)
modelA1 = {}
modelA1['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA1['Psi2'] = np.random.randn(H) / np.sqrt(H)
modelA2 = {}
modelA2['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA2['Psi2'] = np.random.randn(H) / np.sqrt(H)
modelA3 = {}
modelA3['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA3['Psi2'] = np.random.randn(H) / np.sqrt(H)

gradA0_buffer = { k : np.zeros_like(v) for k,v in modelA0.iteritems() } 
gradA1_buffer = { k : np.zeros_like(v) for k,v in modelA1.iteritems() } 
gradA2_buffer = { k : np.zeros_like(v) for k,v in modelA2.iteritems() } 
gradA3_buffer = { k : np.zeros_like(v) for k,v in modelA3.iteritems() } 

rmspropA0_cache = { k : np.zeros_like(v) for k,v in modelA0.iteritems() } # rmsprop memory
rmspropA1_cache = { k : np.zeros_like(v) for k,v in modelA1.iteritems() } 
rmspropA2_cache = { k : np.zeros_like(v) for k,v in modelA2.iteritems() } 
rmspropA3_cache = { k : np.zeros_like(v) for k,v in modelA3.iteritems() } 
 
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

def discount_rewards(r):      # take 1D float array of rewards and compute discounted reward
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary 
    #   R_t     = r_t + running_add * gamma = r_t + gamma * R_t+1
    running_add = r[t] + running_add * gamma  
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(model,x):
  h = np.dot(model['Psi1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['Psi2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(model,eph, epdlogp):
  dW2 = np.dot(eph.T, epdlogp).ravel()    #(200,)    = <(5,200)', (5,1)>
  dh = np.outer(epdlogp, model['Psi2'])     #(5,200)   = (5,1) (200,)
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)           #(200,6400) = <(5,200)',(5,6400)>
  return {'Psi1':dW1, 'Psi2':dW2}

env = gym.make("LunarLander-v2")
x = env.reset()
x_prev = None
xs,hA0s,hA1s,hA2s,hA3s,dlogps0,dlogps1,dlogps2,dlogps3,drs = [],[],[],[],[],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0

while True:
  step_number += 1

  x_prev = x

  # forward the policy network and sample an action from the returned probabilities
  ac_prob0, hA0 = policy_forward(modelA0,x)
  ac_prob1, hA1 = policy_forward(modelA1,x)
  ac_prob2, hA2 = policy_forward(modelA2,x)
  ac_prob3, hA3 = policy_forward(modelA3,x)
  action = int(epsilon_greedy_exploration(np.argmax([ac_prob0,ac_prob1,ac_prob2,ac_prob3]), episode_number))


  # record various intermediates (needed later for backprop) before stepping
  xs.append(x) 
  hA0s.append(hA0) 
  hA1s.append(hA1) 
  hA2s.append(hA2) 
  hA3s.append(hA3) 
  err_prob0 = (1-ac_prob0) if action == 0 else (0-ac_prob0)  #error in probability of expected label
  err_prob1 = (1-ac_prob1) if action == 1 else (0-ac_prob1)
  err_prob2 = (1-ac_prob2) if action == 2 else (0-ac_prob2)
  err_prob3 = (1-ac_prob3) if action == 3 else (0-ac_prob3)
  dlogps0.append(err_prob0)
  dlogps1.append(err_prob1)
  dlogps2.append(err_prob2)
  dlogps3.append(err_prob3)

  x, reward, done, info = env.step(action) 
  reward_sum += reward
  drs.append(reward) 

  if done: 
    episode_number += 1

    epx     = np.vstack(xs)
    ephA0   = np.vstack(hA0s)
    ephA1   = np.vstack(hA1s)
    ephA2   = np.vstack(hA2s)
    ephA3   = np.vstack(hA3s)
    epdlogp0= np.vstack(dlogps0)
    epdlogp1= np.vstack(dlogps1)
    epdlogp2= np.vstack(dlogps2)
    epdlogp3= np.vstack(dlogps3)
    epr     = np.vstack(drs)
    xs,hA0s,hA1s,hA2s,hA3s,dlogps0,dlogps1,dlogps2,dlogps3,drs = [],[],[],[],[],[],[],[],[],[]

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp0 *= discounted_epr 
    grad = policy_backward(modelA0, ephA0, epdlogp0)
    for k in modelA0: gradA0_buffer[k] += grad[k] # accumulate grad over batch
    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in modelA0.iteritems():
        g = gradA0_buffer[k] # gradient
        rmspropA0_cache[k] = decay_rate * rmspropA0_cache[k] + (1 - decay_rate) * g**2
        modelA0[k] += alpha * g / (np.sqrt(rmspropA0_cache[k]) + 1e-5)
        gradA0_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    epdlogp1 *= discounted_epr 
    grad = policy_backward(modelA1, ephA1, epdlogp1)
    for k in modelA1: gradA1_buffer[k] += grad[k] 
    if episode_number % batch_size == 0:
      for k,v in modelA1.iteritems():
        g = gradA1_buffer[k] 
        rmspropA1_cache[k] = decay_rate * rmspropA1_cache[k] + (1 - decay_rate) * g**2
        modelA1[k] += alpha * g / (np.sqrt(rmspropA1_cache[k]) + 1e-5)
        gradA1_buffer[k] = np.zeros_like(v) 

    epdlogp2 *= discounted_epr 
    grad = policy_backward(modelA2, ephA2, epdlogp2)
    for k in modelA2: gradA2_buffer[k] += grad[k] 
    if episode_number % batch_size == 0:
      for k,v in modelA2.iteritems():
        g = gradA2_buffer[k] 
        rmspropA2_cache[k] = decay_rate * rmspropA2_cache[k] + (1 - decay_rate) * g**2
        modelA2[k] += alpha * g / (np.sqrt(rmspropA2_cache[k]) + 1e-5)
        gradA2_buffer[k] = np.zeros_like(v) 

    epdlogp3 *= discounted_epr 
    grad = policy_backward(modelA3, ephA3, epdlogp3)
    for k in modelA3: gradA3_buffer[k] += grad[k] 
    if episode_number % batch_size == 0:
      for k,v in modelA3.iteritems():
        g = gradA3_buffer[k] 
        rmspropA3_cache[k] = decay_rate * rmspropA3_cache[k] + (1 - decay_rate) * g**2
        modelA3[k] += alpha * g / (np.sqrt(rmspropA3_cache[k]) + 1e-5)
        gradA3_buffer[k] = np.zeros_like(v) 


    # book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished. episode:%d, total steps %d, total reward %f. running mean: %f' % (episode_number-1,step_number-1, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset()
    x_prev = None
    step_number=0
  # print ('ep %d: step %d, rwd %f for action %d' % (episode_number,step_number, reward, action))
