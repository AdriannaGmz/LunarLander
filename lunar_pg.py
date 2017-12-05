""" Trains an Actor with CACLA learning through a Critic. Uses OpenAI Gym. """
# Two neural networks are used 
# Actor  is NN1, Psi parameters
# Critic is NN2, Theta parameters

import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200         # number of hidden layer neurons
# batch_size = 10 # every how many episodes to do a param update?
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-4    # learning_rate of actor
beta = 1e-2     # learning_rate of critic
gamma = 0.99    # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False


#models initialization, Actor and Critic
D = 8 # observation space
A = 4 # action space
modelA = {}
modelA['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA['Psi2'] = np.random.randn(A,H) / np.sqrt(H)
modelC = {}
modelC['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
modelC['Theta2'] = np.random.randn(H) / np.sqrt(H)

gradA_buffer = { k : np.zeros_like(v) for k,v in modelA.iteritems() } # update buffers that add up gradients over a batch
gradC_buffer = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 
rmspropA_cache = { k : np.zeros_like(v) for k,v in modelA.iteritems() } # rmsprop memory
rmspropC_cache = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

  
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
  hA = np.dot(modelA['Psi1'], x)
  hA[hA<0] = 0    # ReLU nonlinearity
  logp = np.dot(modelA['Psi2'], hA)  #here is weird.. need to transpose to 4 actions, not only one
  p = sigmoid(logp)
  return p, hA     

def critic_forward(x):
  hC = np.dot(modelC['Theta1'], x)
  hC[hC<0] = 0      
  logv = np.dot(modelC['Theta2'], hC)
  v = sigmoid(logv)
  return v, hC              # v is value function 

# def actor_backward(hA, ac_prob):   #backpropagation also has to be by four, not 1. NOT BY EPISODE!
#   # dPsi2 = np.dot(hA.T, ac_prob).ravel()
#   dPsi2 = np.dot(np.vstack(hA.T), np.vstack(ac_prob).T)  #(200,4)  = <(200,1), (1,4)> 
#   # dhA = np.outer(np.vstack(ac_prob).T, modelA['Psi2'])   #DEBE SER (1,200)  = (1,4) X (4,200)
#   dhA = 0.0;
#   for i in range(len(ac_prob)):
#     dhA += np.outer(ac_prob[i], modelA['Psi2'][i])   #DEBE SER (1,200)  = (1,4) X (4,200) 
#   dhA[np.vstack(hA).T <= 0] = 0 # backpro prelu
#   dPsi1 = np.dot(dhA.T, np.vstack(x).T)                  #DEBE SER (200,8) = < (_1,200_)' (1,8)>
#   return {'Psi1':dPsi1, 'Psi2':dPsi2}
#                 # >>> modelA['Psi1'].shape        # (200, 8)
#                 # >>> modelA['Psi2'].shape        # (4, 200)

def actor_backward(ephA, epdlogp,action):   #backpropagation only for weights that correspond to the performed action
  dPsi2 = np.dot(ephA.T, epdlogp)           #(200,4)'   = <(200,1), (1,4)> 
  dhA   = np.outer(epdlogp.T[action], modelA['Psi2'][action])   #DEBE SER (1,200)  = (1,4) X (4,200) 
  dhA[ephA <= 0] = 0 # backpro prelu
  dPsi1 = np.dot(dhA.T, epx)                  #DEBE SER (200,8) = < (_1,200_)' (1,8)>
  return {'Psi1':dPsi1, 'Psi2':dPsi2.T}
                # >>> modelA['Psi1'].shape        # (200, 8)
                # >>> modelA['Psi2'].shape        # (4, 200)

def critic_backward(ephC, epvs):
  dTheta2 = np.dot(ephC.T, epvs).ravel()         # (200,)   = <(200,)', (1)>
  dhC = np.outer(epvs, modelC['Theta2'])       # (1,200)  =  (1) X (200,) 
  dhC[ ephC <= 0] = 0 # backpro prelu
  dTheta1 = np.dot(dhC.T, epx)   # (200,8) = < (1,200)' (1,8)>
  return {'Theta1':dTheta1, 'Theta2':dTheta2}
                # >>> modelC['Theta1'].shape      # (200, 8)
                # >>> modelC['Theta2'].shape      # (200,)

def discount_rewards(r):      # take 1D float array of rewards and compute discounted reward
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary 
    #   R_t     = r_t + running_add * gamma = r_t + gamma * R_t+1
    running_add = r[t] + running_add * gamma  
    discounted_r[t] = running_add
  return discounted_r


env = gym.make("LunarLander-v2")
x = env.reset()
x_prev = None
xs,hAs,hCs,dlogps,drs,deltas,vs = [],[],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0


while True:
  if render: env.render()
  step_number += 1


  # ALG. Choose a from policy(s,psi):
      # forward the Actor to get actions probabilities
  ac_prob, hA = actor_forward(x)
      # Sample an action from the returned probability with greedy exploration
  action    = int(epsilon_greedy_exploration(np.argmax(ac_prob), episode_number))

  # ALG. Perform a, observe r and s'
  x_prev = x

  # record various intermediates (needed later for backprop) before stepping
  xs.append(x) 
  hAs.append(hA) 
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - ac_prob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  x, reward, done, info = env.step(action) 
  reward_sum += reward
  drs.append(reward) 
  # print ('step %d: reward  %f for action %d' % (step_number, reward, action)) 


  # ALG. calculate delta = r + gamma*V(s') - V(s)
  v_prev, hC_prev = critic_forward(x_prev)
  v, hC           = critic_forward(x)
  hCs.append(hC) 
  delta_t         = reward + gamma*v - v_prev
  vs.append(v) 
  deltas.append(delta_t)

  if done: 
    episode_number += 1

    epx     = np.vstack(xs)
    ephA    = np.vstack(hAs)
    ephC    = np.vstack(hCs)
    epdlogp = np.vstack(dlogps)
    epvs    = np.vstack(vs)
    epr     = np.vstack(drs)
    epdelta = np.vstack(deltas)
    xs,hAs,hCs,dlogps,drs,deltas,vs = [],[],[],[],[],[],[]

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    # UPDATING CRITIC PARAMETERS: THETA
    epvs *= discounted_epr                # multiplying by discounted rwd
    # epvs *= epdelta                # multiplying by deltas
    gradC = critic_backward(ephC, epvs)
    for k in modelC: gradC_buffer[k] += gradC[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in modelC.iteritems():
        g = gradC_buffer[k] # gradient
        rmspropC_cache[k] = decay_rate * rmspropC_cache[k] + (1 - decay_rate) * g**2
        modelC[k] += beta * g / (np.sqrt(rmspropC_cache[k]) + 1e-5)
        gradC_buffer[k] = np.zeros_like(v)          # reset batch gradient buffer


    # UPDATING ACTOR PARAMETERS: PSI, if delta>0
    epdlogp *= discounted_epr       # multiplying by discounted rwd
    # epdlogp *= epdelta                # multiplying by deltas
    # delta_pos = np.vstack(deltas)       
    epdlogp = epdlogp*[epdelta>0]     # IF DELTA positive
    epdlogp = np.squeeze(epdlogp)
    gradA = actor_backward(ephA, epdlogp,action)
    for k in modelA: gradA_buffer[k] += gradA[k] 

    if episode_number % batch_size == 0:
      for k,v in modelA.iteritems():
        g = gradA_buffer[k] # gradient
        rmspropA_cache[k] = decay_rate * rmspropA_cache[k] + (1 - decay_rate) * g**2
        modelA[k] += alpha * g / (np.sqrt(rmspropA_cache[k]) + 1e-5)
        gradA_buffer[k] = np.zeros_like(v)       


    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished, steps: %d, episode %d total reward: %f. running mean: %f' % (step_number,episode_number, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset() # reset env
    x_prev = None
    step_number=0

