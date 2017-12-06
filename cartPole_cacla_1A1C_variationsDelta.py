""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
# A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
# The system is controlled by applying a force of +1 or -1 to the cart. 
# The pendulum starts upright, and the goal is to prevent it from falling over. 
# A reward of +1 is provided for every timestep that the pole remains upright. 
# The episode ends when the pole is more than 15 degrees from vertical, 
# or the cart moves more than 2.4 units from the center.

import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-4
beta = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = True

# model initialization
D = 4 # observation space
A = 2 # action space

modelA = {}
modelA['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA['Psi2'] = np.random.randn(A,H) / np.sqrt(H)

modelC = {}
modelC['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
modelC['Theta2'] = np.random.randn(H) / np.sqrt(H)

gradA_buffer = { k : np.zeros_like(v) for k,v in modelA.iteritems() } 
gradC_buffer = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

rmspropA_cache = { k : np.zeros_like(v) for k,v in modelA.iteritems() } 
rmspropC_cache = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 


  
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



def actor_forward(x):
  hA = np.dot(modelA['Psi1'], x)
  hA[hA<0] = 0    # ReLU nonlinearity
  logp = np.dot(modelA['Psi2'], hA)  
  p = sigmoid(logp)
  return p, hA  

def critic_forward(x):
  hC = np.dot(modelC['Theta1'], x)
  hC[hC<0] = 0      
  logv = np.dot(modelC['Theta2'], hC)
  v = sigmoid(logv)
  return v, hC              
  #return logv, hC              # logv is the value for the value function. Linear activation

def actor_backward(hA, err_probs):   
  dPsi2 = np.outer(hA.T, err_probs).T            #(200,4)'   = <(200,1)', (4,)> '
  dhA  = np.dot(err_probs.T, modelA['Psi2']).T   #(1,200)'  = (4,)' (4,200) 
  dhA[hA <= 0] = 0                               # backpro prelu
  dPsi1 = np.dot(np.vstack(dhA), np.vstack(x).T) #(200,8) = < (_1,200_)' (1,8)>
  return {'Psi1':dPsi1, 'Psi2':dPsi2}
                # >>> modelA['Psi1'].shape        # (200, 8)
                # >>> modelA['Psi2'].shape        # (4,200)

def critic_backward(hC, v):
  dTheta2 = v*hC                              # (200,)   = <(200,)', (1)>
  dhC = v*modelC['Theta2']                     # (1,200)  =  (1) X (200,) 
  dhC[hC <= 0] = 0                             # backpro prelu
  dTheta1 = np.dot(np.vstack(dhC),np.vstack(x).T)      # (200,8) = < (1,200)' (1,8)>
  return {'Theta1':dTheta1, 'Theta2':dTheta2}
                # >>> modelC['Theta1'].shape      # (200, 8)
                # >>> modelC['Theta2'].shape      # (200,)

running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0
x_prev = None
err_probs = np.zeros(A);


env = gym.make('CartPole-v0')
x = env.reset()
x_prev = None


while True:
  if render: env.render()
  step_number += 1
  x_prev = x

  ac_prob, hA = actor_forward(x)
  action = sample_from_action_probs(ac_prob)

  # ALG. Perform a, observe r and s'
  x_prev = x
  x, reward, done, info = env.step(action) 
  reward_sum += reward

  v_prev, hC_prev = critic_forward(x_prev)
  v, hC           = critic_forward(x)
  delta_t         = reward + gamma*v - v_prev

  err_v = (v - v_prev)**2 if delta_t>0 else 10000 # a big mistake
  grad_C = critic_backward(hC, err_v)

  for k in modelC: gradC_buffer[k] += grad_C[k] 
  if step_number % batch_size == 0:
    for k,v in modelC.iteritems():
      g = gradC_buffer[k] 
      rmspropC_cache[k] = decay_rate * rmspropC_cache[k] + (1 - decay_rate) * g**2
      modelC[k] += - beta * g / (np.sqrt(rmspropC_cache[k]) + 1e-5)
      gradC_buffer[k] = np.zeros_like(v) 
  if delta_t>0:
    #     ALG. Psi_t = Psi_t + alpha*(a - Ac(s,psi)) grad_Ac(s,psi)
    for k in range(len(err_probs)):
      err_probs[k] = (1-ac_prob[k])**2 if k == action else (0-ac_prob[k])**2  #error in probability of expected label

    # only for the non-executed actions
      # modelA[k] += alpha * (action - ac_prob[action])* grad_A[k]
      grad_A = actor_backward(hA,err_probs)
      # for k,v in modelA.iteritems():   
      #   modelA[k] += -alpha * grad_A[k]
      for k in modelA: gradA_buffer[k] += grad_A[k] 
      if step_number % batch_size == 0:
        for k,v in modelA.iteritems():
          g = gradA_buffer[k] 
          rmspropA_cache[k] = decay_rate * rmspropA_cache[k] + (1 - decay_rate) * g**2
          modelA[k] += - alpha * g / (np.sqrt(rmspropA_cache[k]) + 1e-5)
          gradA_buffer[k] = np.zeros_like(v) 


  if done: # an episode finished
    episode_number += 1


    # book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished. episode:%d, total steps %d, total reward %f. running mean: %f' % (episode_number-1,step_number-1, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset()
    x_prev = None
    step_number=0
  # print ('ep %d: step %d, rwd %f for action %d' % (episode_number,step_number, reward, action))
