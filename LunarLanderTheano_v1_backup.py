
import numpy as np
import cPickle as pickle
import gym
import theano.tensor as T
import theano



#hyperparameters
H1 = 200         # number of hidden layer neurons
H2 = 133
batch_size = 10 # every how many episodes to do a param update?
alpha = 1e-4    # learning_rate of actor
beta = 1e-2     # learning_rate of critic
gamma = 0.99    # discount factor for reward
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
epsilon_min = 0.5
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



# define the weights as shared variables
weightA_h1  = theano.shared(np.random.randn(D,H1) / np.sqrt(D))
weightA_out = theano.shared(np.random.randn(H1,A) / np.sqrt(H1))

weightA = {}
weightA['Psi1'] = weightA_h1
weightA['Psi2'] = weightA_out

weightC_h1  = theano.shared(0*np.random.randn(D,H1) / np.sqrt(D))
weightC_out = theano.shared(0*np.random.randn(H1,1) / np.sqrt(H1))
weightC = {}
weightC['theta1'] = weightC_h1
weightC['theta2'] = weightC_out


x = theano.shared(x)


# Actor models and functions
actor_hidden1 = x.dot(weightA['Psi1'])
actor_hidden1_sig = 1/(1+ T.exp(-actor_hidden1))
actor_output = actor_hidden1_sig.dot(weightA['Psi2'])
actor_output = 1/(1+ T.exp(-actor_output))

actor_forward = theano.function([], actor_output)
actual_output_A = actor_forward()  #Actual output


actual_output_A = theano.shared(actual_output_A)
actor_cost = -(actual_output_A*T.log(actor_output) + (1-actual_output_A)*T.log(1-actor_output)).sum()
#actor_cost = actual_output_A - actor_output
dw1A,dw2A = T.grad(actor_cost,[weightA['Psi1'],weightA['Psi2']])

# To update the actor taken actual probability values
Y = T.vector()
X_update = (actual_output_A, T.set_subtensor(actual_output_A[[0,1,2,3]], Y))
actual_output_A_update_func = theano.function([Y], updates=[X_update])
actor_gradient_step = theano.function(
    [],
    # profile=True,
    updates=((weightA['Psi1'], weightA['Psi1'] + actor_step * dw1A),
             (weightA['Psi2'], weightA['Psi2'] + actor_step * dw2A)))



# Critic model and functions
critic_hidden1 = x.dot(weightC['theta1'])
#critic_hidden1_sig = 1/(1+ T.exp(-critic_hidden1))
critic_output = critic_hidden1.dot(weightC['theta2'])
critic_output = 1/(1+ T.exp(-critic_output))


#we removed the input values because we will always use the same shared variable
critic_forward = theano.function([], critic_output)
actual_output_C = critic_forward()
actual_output_C = theano.shared(actual_output_C)
critic_cost = -(actual_output_C*T.log(critic_output) + (1-actual_output_C)*T.log(1-critic_output)).sum()
#critic_cost = actual_output_C - critic_output
dw1C,dw2C = T.grad(critic_cost,[weightC_h1,weightC_out])

critic_gradient_step = theano.function(
    [],
    # profile=True,
    updates=((weightC['theta1'], weightC['theta1'] + actor_step * dw1C),
             (weightC['theta2'], weightC['theta2'] + actor_step * dw2C)))


env = gym.make("LunarLander-v2")
x = env.reset()
x_prev = None

while True:
    step_number += 1

    actor_output_A = actor_forward()
    critic_output_C = critic_forward()

    best_action = np.argmax(actor_output_A)


    # choose action and perform it
    action = epsilon_greedy_exploration(np.argmax(actor_output_A), episode_number)
    x_prev = x
    x, reward, done, info = env.step(action)
    reward_sum += reward


    actual_output_C = reward + gamma*critic_forward()

    # updte the actual_output_A according to the chosen action
    for k in range(A):
        actual_output_A_updater[k] = 1 if k == action else 0
    actual_output_A_update_func(actual_output_A_updater)

    critic_gradient_step()
    actor_gradient_step()

    #print(critic_output.get_value())
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
