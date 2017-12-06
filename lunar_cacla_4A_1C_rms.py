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
gamma = 0.99    # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False


#models initialization, Actor and Critic
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

modelC = {}
modelC['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
modelC['Theta2'] = np.random.randn(H) / np.sqrt(H)

gradA0_buffer = { k : np.zeros_like(v) for k,v in modelA0.iteritems() } 
gradA1_buffer = { k : np.zeros_like(v) for k,v in modelA1.iteritems() } 
gradA2_buffer = { k : np.zeros_like(v) for k,v in modelA2.iteritems() } 
gradA3_buffer = { k : np.zeros_like(v) for k,v in modelA3.iteritems() } 
gradC_buffer = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

rmspropA0_cache = { k : np.zeros_like(v) for k,v in modelA0.iteritems() } # rmsprop memory
rmspropA1_cache = { k : np.zeros_like(v) for k,v in modelA1.iteritems() } 
rmspropA2_cache = { k : np.zeros_like(v) for k,v in modelA2.iteritems() } 
rmspropA3_cache = { k : np.zeros_like(v) for k,v in modelA3.iteritems() } 
rmspropC_cache = { k : np.zeros_like(v) for k,v in modelC.iteritems() } 

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

def critic_forward(x):
  hC = np.dot(modelC['Theta1'], x)
  hC[hC<0] = 0      
  logv = np.dot(modelC['Theta2'], hC)
  # v = sigmoid(logv)
  # return v, hC              
  return logv, hC              # logv is the value for the value function. Linear activation

def actor_backward(model, hA, err_ac):   
  dPsi2 = np.dot(hA.T, err_ac)                     #(200,)   = <(200,)', (1)> 
  dhA  = err_ac*model['Psi2']                    #(1,200)  = (1) X (200) 
  dhA[np.vstack(hA).T <= 0] = 0                  # backpro prelu
  dPsi1 = np.dot(dhA.T, np.vstack(x).T)          #(200,8) = < (_1,200_)' (1,8)>
  return {'Psi1':dPsi1, 'Psi2':dPsi2}
                # >>> modelA['Psi1'].shape        # (200, 8)
                # >>> modelA['Psi2'].shape        # (200,)

def critic_backward(hC, v):
  dTheta2 = np.dot(hC.T, v)                      # (200,)   = <(200,)', (1)>
  dhC = v*modelC['Theta2']                     # (1,200)  =  (1) X (200,) 
  dhC[np.vstack(hC).T <= 0] = 0                # backpro prelu
  dTheta1 = np.dot(dhC.T, np.vstack(x).T)      # (200,8) = < (1,200)' (1,8)>
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
x_prev = None

while True:
  # if render: env.render()
  step_number += 1

  # ALG. Choose a from policy(s,psi):
      # forward the Actor to get actions probabilities
  ac_prob0, hA0 = actor_forward(modelA0,x)
  ac_prob1, hA1 = actor_forward(modelA1,x)
  ac_prob2, hA2 = actor_forward(modelA2,x)
  ac_prob3, hA3 = actor_forward(modelA3,x)
      # Sample an action from the returned probabilities with greedy exploration
  action = int(epsilon_greedy_exploration(np.argmax([ac_prob0,ac_prob1,ac_prob2,ac_prob3]), episode_number))

  # ALG. Perform a, observe r and s'
  x_prev = x
  x, reward, done, info = env.step(action) 
  reward_sum += reward
  drs.append(reward)
  # print ('step %d: reward  %f for action %d' % (step_number, reward, action)) 


  # ALG. calculate delta = r + gamma*V(s') - V(s)
  v_prev, hC_prev = critic_forward(x_prev)
  v, hC           = critic_forward(x)
  delta_t         = reward + gamma*v - v_prev


  #   ALG. Theta_t = Theta_t + beta*delta*gradient_V(s)  BACKPROPAGATION CRITIC
  # err_v = reward + gamma*v 
  err_v = v - v_prev
  grad_C = critic_backward(hC, err_v)
  for k in modelC: gradC_buffer[k] += grad_C[k] 
  if step_number % batch_size == 0:
    for k,v in modelC.iteritems():
      rmspropC_cache[k] = decay_rate * rmspropC_cache[k]+  (1 - decay_rate) * grad_C[k]**2
      modelC[k] += beta * grad_C[k] / (np.sqrt(rmspropC_cache[k]) + 1e-5)
      # modelC[k] += beta * delta_t * grad_C[k]
    

  #   ALG. if delta > 0 then
  if delta_t>0:
    #     ALG. Psi_t = Psi_t + alpha*(a - Ac(s,psi)) grad_Ac(s,psi)

    err_prob0 = (1-ac_prob0) if action == 0 else (0-ac_prob0)  #error in probability of expected label
    err_prob1 = (1-ac_prob1) if action == 1 else (0-ac_prob1)
    err_prob2 = (1-ac_prob2) if action == 2 else (0-ac_prob2)
    err_prob3 = (1-ac_prob3) if action == 3 else (0-ac_prob3)

    # only for the non-executed actions
        # modelA[k] += alpha * (action - ac_prob[action])* grad_A[k]
    if action!=0:
      grad_A0 = actor_backward(modelA0,hA0,err_prob0)
      # for k,v in modelA0.iteritems():   
      #   modelA0[k] += alpha * grad_A0[k]
      for k in modelA0: gradA0_buffer[k] += grad_A0[k] 
      if step_number % batch_size == 0:
        for k,v in modelA0.iteritems():
          g = gradA0_buffer[k] # gradient
          rmspropA0_cache[k] = decay_rate * rmspropA0_cache[k] + (1 - decay_rate) * g**2
          modelA0[k] += alpha * g / (np.sqrt(rmspropA0_cache[k]) + 1e-5)
          gradA0_buffer[k] = np.zeros_like(v)       



    if action!=1:
      grad_A1 = actor_backward(modelA1,hA1,err_prob1)
      # for k,v in modelA1.iteritems():   
      #   modelA1[k] += alpha * grad_A1[k]
      for k in modelA1: gradA1_buffer[k] += grad_A1[k] 
      if step_number % batch_size == 0:
        for k,v in modelA1.iteritems():
          g = gradA1_buffer[k] # gradient
          rmspropA1_cache[k] = decay_rate * rmspropA1_cache[k] + (1 - decay_rate) * g**2
          modelA1[k] += alpha * g / (np.sqrt(rmspropA1_cache[k]) + 1e-5)
          gradA1_buffer[k] = np.zeros_like(v)       


    if action!=2:
      grad_A2 = actor_backward(modelA2,hA2,err_prob2)
      # for k,v in modelA2.iteritems():   
      #   modelA2[k] += alpha * grad_A2[k]
      for k in modelA2: gradA2_buffer[k] += grad_A2[k] 
      if step_number % batch_size == 0:
        for k,v in modelA2.iteritems():
          g = gradA2_buffer[k] # gradient
          rmspropA2_cache[k] = decay_rate * rmspropA2_cache[k] + (1 - decay_rate) * g**2
          modelA2[k] += alpha * g / (np.sqrt(rmspropA2_cache[k]) + 1e-5)
          gradA2_buffer[k] = np.zeros_like(v)       


    if action!=3:
      grad_A3 = actor_backward(modelA3,hA3,err_prob3)
      # for k,v in modelA0.iteritems():   
      #   modelA3[k] += alpha * grad_A3[k]
      for k in modelA3: gradA3_buffer[k] += grad_A3[k] 
      if step_number % batch_size == 0:
        for k,v in modelA3.iteritems():
          g = gradA3_buffer[k] # gradient
          rmspropA3_cache[k] = decay_rate * rmspropA3_cache[k] + (1 - decay_rate) * g**2
          modelA3[k] += alpha * g / (np.sqrt(rmspropA3_cache[k]) + 1e-5)
          gradA3_buffer[k] = np.zeros_like(v)       





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
