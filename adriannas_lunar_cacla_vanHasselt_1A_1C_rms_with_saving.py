""" Trains an Actor with CACLA learning through a Critic. Uses OpenAI Gym. """
# 2 neural networks are used, 200 neurons for the single hidden layer 
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
resume = True


#models initialization, Actor and Critic
D = 8 # observation space
A = 4 # action space

if resume:

  modelA = pickle.load(open('saveA.p', 'rb'))
  modelC = pickle.load(open('saveC.p', 'rb'))

  gradA_buffer = pickle.load(open('saveGradA.p', 'rb'))
  gradC_buffer = pickle.load(open('saveGradC.p', 'rb'))

  rmspropA_cache = pickle.load(open('saveRmsA.p', 'rb'))
  rmspropC_cache = pickle.load(open('saveRmsC.p', 'rb'))

else:

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


running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0
err_probs = np.zeros(A);

  
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
  

def actor_backward(x,target_a):
  ac_prob, hA = actor_forward(x)
  # error = target_a - ac_prob
  error = target_a
  dPsi2 = np.outer(hA.T, error).T            #(200,4)'   = <(200,1)', (4,)> '
  dhA  = np.dot(error.T, modelA['Psi2']).T   #(1,200)'  = (4,)' (4,200) 
  dhA[hA <= 0] = 0                               # backpro prelu
  dPsi1 = np.dot(np.vstack(dhA), np.vstack(x).T) #(200,8) = < (_1,200_)' (1,8)>
  return {'Psi1':dPsi1, 'Psi2':dPsi2}
                # >>> modelA['Psi1'].shape        # (200, 8)
                # >>> modelA['Psi2'].shape        # (4,200)

def critic_backward(x,target_v):
  v, hC = critic_forward(x)
  dTheta2 = (target_v-v)*hC                              # (200,)   = <(200,)', (1)>
  dhC = target_v*modelC['Theta2']                     # (1,200)  =  (1) X (200,) 
  dhC[hC <= 0] = 0                             # backpro prelu
  dTheta1 = np.dot(np.vstack(dhC),np.vstack(x).T)      # (200,8) = < (1,200)' (1,8)>
  return {'Theta1':dTheta1, 'Theta2':dTheta2}
                # >>> modelC['Theta1'].shape      # (200, 8)
                # >>> modelC['Theta2'].shape      # (200,)

env = gym.make("LunarLander-v2")
x = env.reset()
x_prev = None

while True:
  step_number += 1

  # ALG. Choose a from policy(s,psi):
      # forward the Actor to get actions probabilities
  ac_prob, hA = actor_forward(x)
      # Sample an action from the returned probabilities with greedy exploration
  action = sample_from_action_probs(ac_prob)

  # ALG. Perform a, observe r and s'
  x_prev = x
  hA_prev = hA
  x, reward, done, info = env.step(action) 
  reward_sum += reward


  # ALG. calculate delta = r + gamma*V(s') - V(s)
  v_x, hC          = critic_forward(x)
  v_target         = reward + gamma*v_x


  #   ALG. Theta_t = Theta_t + beta*delta*gradient_V(s)  BACKPROPAGATION CRITIC
  # err_v = reward + gamma*v 
  # err_v = v - v_prev if v_target>0 else 10000 # a big mistake
  grad_C = critic_backward(x_prev, v_target)
  # for k,v in modelC.iteritems():
  #   modelC[k] += beta * grad_C[k] 
  for k in modelC: gradC_buffer[k] += grad_C[k] 
  if step_number % batch_size == 0:
    for k,v in modelC.iteritems():
      g = gradC_buffer[k] 
      rmspropC_cache[k] = decay_rate * rmspropC_cache[k] + (1 - decay_rate) * g**2
      modelC[k] += - beta * g / (np.sqrt(rmspropC_cache[k]) + 1e-5)
      gradC_buffer[k] = np.zeros_like(v) 


    

  #   ALG. if delta > 0 then
  if v_target > v_x:
    #     ALG. Psi_t = Psi_t + alpha*(a - Ac(s,psi)) grad_Ac(s,psi)
    for k in range(len(err_probs)):
      err_probs[k] = (1-ac_prob[k]) if k == action else (0-ac_prob[k])  #error in probability of expected label
      # ac_target[k] = 1 if k == action else (0-ac_prob[k])

  # only for the non-executed actions
    # modelA[k] += alpha * (action - ac_prob[action])* grad_A[k]
    grad_A = actor_backward(x_prev,err_probs)
    # grad_A = actor_backward(x,ac_target)
    # for k,v in modelA.iteritems():   
      # modelA[k] += alpha * grad_A[k]
    for k in modelA: gradA_buffer[k] += grad_A[k] 
    if step_number % batch_size == 0:
      for k,v in modelA.iteritems():
        g = gradA_buffer[k] 
        rmspropA_cache[k] = decay_rate * rmspropA_cache[k] + (1 - decay_rate) * g**2
        modelA[k] += - alpha * g / (np.sqrt(rmspropA_cache[k]) + 1e-5)
        gradA_buffer[k] = np.zeros_like(v) 




  if done: # an episode finished
    episode_number += 1

    if episode_number % 100 == 0: 
      pickle.dump(modelA, open('saveA.p', 'wb'))
      pickle.dump(modelC, open('saveC.p', 'wb'))

      pickle.dump(gradA_buffer, open('saveGradA.p', 'wb'))
      pickle.dump(gradC_buffer, open('saveGradC.p', 'wb'))

      pickle.dump(rmspropA_cache, open('saveRmsA.p', 'wb'))
      pickle.dump(rmspropC_cache, open('saveRmsC.p', 'wb'))


    # book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished. episode:%d, total steps %d, total reward %f. running mean: %f' % (episode_number-1,step_number-1, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset()
    x_prev = None
    step_number=0
  # print ('ep %d: step %d, rwd %f for action %d' % (episode_number,step_number, reward, action))
