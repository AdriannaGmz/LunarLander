""" Trains an Actor with CACLA learning through a Critic. Uses OpenAI Gym. """
# 5 neural networks are used, 200 neurons for the single hidden layer 
# Actor  is Psi parameters, one NN for each action. 4 actions = 0,1, 2, 3. Sigmoid activation function
# Critic is Theta parameters, one NN. Linear activaton function

import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200         # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-4    # learning_rate of actor
beta = 1e-2     # learning_rate of critic
# alpha = 0.5    # learning_rate of actor
# beta = 0.5     # learning_rate of critic
# gamma = 0.99    # discount factor for reward
gamma = 0.2    # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False


#models initialization, Actor and Critic
D = 8 # observation space
A = 4 # action space
modelA0 = {}
modelA0['Psi1'] = np.random.randn(H,D) / np.sqrt(D)
modelA0['Psi2'] = np.random.randn(H) / np.sqrt(H)
modelA1 = { k : np.zeros_like(v) for k,v in modelA0.iteritems() }
modelA2 = { k : np.zeros_like(v) for k,v in modelA0.iteritems() }
modelA3 = { k : np.zeros_like(v) for k,v in modelA0.iteritems() }

modelC = {}
modelC['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
modelC['Theta2'] = np.random.randn(H) / np.sqrt(H)

rmspropA0_cache = { k : np.zeros_like(v) for k,v in modelA0.iteritems() } # rmsprop memory
rmspropA1_cache = { k : np.zeros_like(v) for k,v in modelA1.iteritems() } 
rmspropA2_cache = { k : np.zeros_like(v) for k,v in modelA2.iteritems() } 
rmspropA3_cache = { k : np.zeros_like(v) for k,v in modelA3.iteritems() } 
rmspropC_cache = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

# xs,hAs,hCs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0
x_prev = None

  
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




def actor_forward(model,x):
  hA = np.dot(model['Psi1'], x)
  hA[hA<0] = 0    # ReLU nonlinearity
  logp = np.dot(model['Psi2'], hA)  
  p = sigmoid(logp)
  return p, hA  

# def actor0_forward(x):
#   hA0 = np.dot(modelA0['Psi1'], x)
#   hA0[hA0<0] = 0    # ReLU nonlinearity
#   logp = np.dot(modelA0['Psi2'], hA0)  
#   p = sigmoid(logp)
#   return p, hA0     

# def actor1_forward(x):
#   hA1 = np.dot(modelA1['Psi1'], x)
#   hA1[hA1<0] = 0    
#   logp = np.dot(modelA1['Psi2'], hA1)
#   p = sigmoid(logp)
#   return p, hA1

# def actor2_forward(x):
#   hA2 = np.dot(modelA2['Psi1'], x)
#   hA2[hA2<0] = 0    
#   logp = np.dot(modelA2['Psi2'], hA2)
#   p = sigmoid(logp)
#   return p, hA2

# def actor3_forward(x):
#   hA3 = np.dot(modelA3['Psi1'], x)
#   hA3[hA3<0] = 0    
#   logp = np.dot(modelA3['Psi2'], hA3)
#   p = sigmoid(logp)
#   return p, hA3



def critic_forward(x):
  hC = np.dot(modelC['Theta1'], x)
  hC[hC<0] = 0      
  logv = np.dot(modelC['Theta2'], hC)
  # v = sigmoid(logv)
  # return v, hC              # v is value function 
  return logv, hC              # logv is the value for the value function. Linear activation

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


# def actor_backward(model, hA, ac_prob,action):   #backpropagation only for weights that correspond to the performed action
#   dPsi2 = np.dot(np.vstack(hA.T), np.vstack(ac_prob).T).T  #(200,4)'   = <(200,1), (1,4)> 
#   dhA  = np.outer(ac_prob[action], modelA['Psi2'][action])   #DEBE SER (1,200)  = (1,4) X (4,200) 
#   dhA[np.vstack(hA).T <= 0] = 0 # backpro prelu
#   dPsi1 = np.dot(dhA.T, np.vstack(x).T)                  #DEBE SER (200,8) = < (_1,200_)' (1,8)>
#   return {'Psi1':dPsi1, 'Psi2':dPsi2}
#                 # >>> modelA['Psi1'].shape        # (200, 8)
#                 # >>> modelA['Psi2'].shape        # (4, 200)





def actor_backward(model, hA, err_ac):   
  dPsi2 = np.dot(hA.T, err_ac).ravel()   #(200,)   = <(200,)', (1)> 
  dhA  = np.outer(err_ac, model['Psi2'])   #DEBE SER (1,200)  = (1,4) X (4,200) 
  dhA[np.vstack(hA).T <= 0] = 0 # backpro prelu
  dPsi1 = np.dot(dhA.T, np.vstack(x).T)    #DEBE SER (200,8) = < (_1,200_)' (1,8)>
  return {'Psi1':dPsi1, 'Psi2':dPsi2}
                # >>> modelA['Psi1'].shape        # (200, 8)
                # >>> modelA['Psi2'].shape        # (4, 200)




def critic_backward(hC, v):
  dTheta2 = np.dot(hC.T, v).ravel()         # (200,)   = <(200,)', (1)>
  dhC = np.outer(v, modelC['Theta2'])       # (1,200)  =  (1) X (200,) 
  dhC[np.vstack(hC).T <= 0] = 0 # backpro prelu
  dTheta1 = np.dot(dhC.T, np.vstack(x).T)   # (200,8) = < (1,200)' (1,8)>
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


drs = []
env = gym.make("LunarLander-v2")
x = env.reset()


while True:
  if render: env.render()
  step_number += 1

  # ALG. Choose a from policy(s,psi):
      # forward the Actor to get actions probabilities
  ac_prob0, hA0 = actor0_forward(x)
  ac_prob1, hA1 = actor1_forward(x)
  ac_prob2, hA2 = actor2_forward(x)
  ac_prob3, hA3 = actor3_forward(x)
      # Sample an action from the returned probability with greedy exploration
  action    = int(epsilon_greedy_exploration(np.argmax(ac_prob), episode_number))

  # ALG. Perform a, observe r and s'
  x_prev = x
  x, reward, done, info = env.step(action) 
  reward_sum += reward
  drs.append(reward)      # record reward (has to be done after we call step() to get reward for previous action)
  # print ('step %d: reward  %f for action %d' % (step_number, reward, action)) 


  # ALG. calculate delta = r + gamma*V(s') - V(s)
  v_prev, hC_prev = critic_forward(x_prev)
  v, hC           = critic_forward(x)
  delta_t         = reward + gamma*v - v_prev

  #   ALG. Theta_t = Theta_t + beta*delta*gradient_V(s)  BACKPROPAGATION CRITIC
  grad_C = critic_backward(hC, v)
  for k,v in modelC.iteritems():
    # rmspropC_cache[k] = decay_rate * rmspropC_cache[k] + (1 - decay_rate) * grad_C[k]**2
    # modelC[k] += beta * grad_C[k] / (np.sqrt(rmspropC_cache[k]) + 1e-5)
    modelC[k] += beta * delta_t * grad_C[k]
    

  #   ALG. if delta > 0 then
  if delta_t>0:
    #     ALG. Psi_t = Psi_t + alpha*(a - Ac(s,psi)) grad_Ac(s,psi)
    expected_prob = 1 if action ==
    error_prob = 
    grad_A = actor_backward(hA, ac_prob,action)
    # only for the executed action
    for k,v in modelA.iteritems():   
      # modelA[k] += alpha * (action - ac_prob[action])* grad_A[k]
      modelA[k] += alpha * grad_A[k]

  # ALG. end if
  # ALG. If s' is terminal then
  #   ALG   s ~ I 
  # ALG. else
  #   ALG. s =s'
  # ALG. end if 


  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epr = np.vstack(drs)
    drs = [] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished, steps: %d, episode %d total reward: %f. running mean: %f' % (step_number,episode_number, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset() # reset env
    x_prev = None
    step_number=0

