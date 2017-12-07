
import numpy as np
import cPickle as pickle
import gym
import theano.tensor as T
import theano



#hyperparameters
H1 = 250         # number of hidden layer neurons
H2 = 130
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-2    # learning_rate of actor
beta = 1e-2     # learning_rate of critic
gamma = 0.9    # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False


#models initialization, Actor and Critic
D = 8 # observation space
A = 4 # action space

# Initialize the environment
env = gym.make("LunarLander-v2")
x = env.reset()
x_prev = None

running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0
epsilon_scale = 1.0
epsilon_min = 0.09
x_prev = None
err_probs = np.zeros(A);
actual_output_A_updater = np.zeros(A)
actor_step = 0.001



def epsilon_greedy_exploration(best_action, episode_number):
  epsilon = epsilon_scale/(epsilon_scale + episode_number)
  if epsilon < epsilon_min:
      epsilon = epsilon_min
  prob_vector = [epsilon/4, epsilon/4, epsilon/4, epsilon/4]
  prob_vector[best_action] = epsilon/4 + 1 - epsilon
  action_to_explore = int(np.random.choice(4, 1, True, prob_vector))
  return action_to_explore

def epsilon_greedy_exploration2(best_action, step_number):
    epsilon = epsilon_scale/(epsilon_scale + 0.001*step_number)
    if epsilon < epsilon_min:
        epsilon = epsilon_min
    temp = np.random.rand()
    prob_vector = [1/4, 1/4, 1/4, 1/4]
    if temp < epsilon:
        action_to_explore = int(np.random.choice(4, 1,prob_vector))
    else:
        action_to_explore = best_action
    return action_to_explore


# define the weights as shared variables
weightA_h1  = theano.shared(np.random.randn(D,H1) )#/ np.sqrt(D))
weightA_h2  = theano.shared(np.random.randn(H1,H2))#/np.sqrt(H2)
weightA_out = theano.shared(np.random.randn(H2,A) )#/ np.sqrt(H2))

weightA = {}
weightA['Psi1'] = weightA_h1
weightA['Psi12']= weightA_h2
weightA['Psi2'] = weightA_out

weightC_h1  = theano.shared(np.random.randn(D,H1) / np.sqrt(D))
weightC_out = theano.shared(np.random.randn(H1,1) / np.sqrt(H1))
weightC = {}
weightC['theta1'] = weightC_h1
weightC['theta2'] = weightC_out


x = theano.shared(x)


# Actor models and functions
actor_hidden1 = x.dot(weightA['Psi1'])
actor_hidden1_sig = 1/(1+ T.exp(-actor_hidden1))
actor_hidden2 = actor_hidden1_sig.dot(weightA['Psi12'])
actor_hidden2_sig = 1/(1+ T.exp(-actor_hidden2))
actor_output = actor_hidden2_sig.dot(weightA['Psi2'])
actor_output = 1/(1+ T.exp(-actor_output))

actor_forward = theano.function([], actor_output)
actual_output_A = T.vector()  #Actual output

actor_cost = -(actual_output_A*T.log(actor_output) + (1-actual_output_A)*T.log(1-actor_output)).sum()
#actor_cost = (actual_output_A - actor_output).sum()
#actor_cost = (actor_output - actual_output_A).sum()
dw1A,dw12A, dw2A = T.grad(actor_cost,[weightA['Psi1'],weightA['Psi12'],weightA['Psi2']])

# To update the actor taken actual probability values
#Y = T.vector()
#X_update = (actual_output_A, T.set_subtensor(actual_output_A[[0,1,2,3]], Y))
#actual_output_A_update_func = theano.function([Y], updates=[X_update])
action = T.scalar() # initialization
best_action = T.scalar()
actor_gradient_step = theano.function(
    [actual_output_A, action, best_action],
    # profile=True,
    updates=((weightA['Psi1'], weightA['Psi1'] + beta * (action - best_action)* dw1A),
             (weightA['Psi12'], weightA['Psi12']+ beta * (action - best_action)* dw12A),
             (weightA['Psi2'], weightA['Psi2'] + beta * (action - best_action)* dw2A)))



# Critic model and functions
critic_hidden1 = x.dot(weightC['theta1'])
critic_hidden1_sig = 1/(1+ T.exp(-critic_hidden1))
critic_output = critic_hidden1.dot(weightC['theta2'])
#critic_output = 1/(1+ T.exp(-critic_output))


#we removed the input values because we will always use the same shared variable
critic_forward = theano.function([], critic_output)
actual_output_C = T.vector()#critic_forward()
#actual_output_C = theano.shared(actual_output_C)
#critic_cost = -(actual_output_C*T.log(critic_output) + (1 - actual_output_C)*T.log(1 - critic_output)).sum()
#critic_cost = T.sum((actual_output_C - critic_output)**1)/1e3
critic_cost = T.sum((actual_output_C - critic_output))
dw1C,dw2C = T.grad(critic_cost,[weightC['theta1'],weightC['theta2']])
delta_t = T.scalar() # initialization
critic_gradient_step = theano.function(
    [actual_output_C],
    # profile=True,
    updates=((weightC['theta1'], weightC['theta1'] + alpha* dw1C),
             (weightC['theta2'], weightC['theta2'] + alpha* dw2C)))


#env = gym.make("LunarLander-v2")
#x = env.reset()
#x_prev = None

while True:
    step_number += 1


    actor_output_A = actor_forward()
    critic_output_C = critic_forward()

    best_action = np.argmax(actor_output_A)
    #critic_out_val.set_value(critic_output_C)

    # choose action and perform it
    action = epsilon_greedy_exploration2(np.argmax(actor_output_A), episode_number)
    x_prev = x
    x, reward, done, info = env.step(action)
    reward_sum += reward

    delta_t = reward + gamma*critic_forward() - critic_output_C
    delta_t = np.asscalar(delta_t) # convert to scalar (it was an array of size 1)
    actual_output_C = reward + gamma*critic_forward()
    #actual_output_C.set_value(reward + gamma*critic_forward())

    # updte the actual_output_A according to the chosen action
    for k in range(A):
        actual_output_A_updater[k] = 1 if k == action else 0
    actual_output_A = actual_output_A_updater
    if delta_t > 0:
        actor_gradient_step(actual_output_A,action, best_action)

    critic_gradient_step(actual_output_C)

    #print(delta_t)
    if done: # an episode finished
        episode_number += 1




        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'game finnished. episode:%d, total steps %d, total reward %f. running mean: %f' % (episode_number-1,step_number-1, reward_sum, running_reward)
        reward_sum = 0
        x = env.reset()
        x = theano.shared(x)
        x_prev = None
        step_number=0
        #print ('ep %d: step %d, rwd %f for action %d' % (episode_number,step_number, reward, action))
