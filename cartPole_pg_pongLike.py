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
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

# model initialization
D = 4 # input dimensionality: 4 grid
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()    #(200,)    = <(5,200)', (5,1)>
  dh = np.outer(epdlogp, model['W2'])     #(5,200)   = (5,1) (200,)
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)           #(200,6400) = <(5,200)',(5,6400)>
  return {'W1':dW1, 'W2':dW2}
# >>> model['W1'].shape     # (200, 6400)
# >>> model['W2'].shape     # (200,)


env = gym.make('CartPole-v0')
x = env.reset()
x_prev = None
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
step_number = 0


while True:
  if render: env.render()
  step_number += 1
  x_prev = x

  aprob, h = policy_forward(x)
  action = 0 if np.random.uniform() < aprob else 1 # roll the dice!
  # action 0, 1 (left and right)

  xs.append(x) 
  hs.append(h) 
  y = 1 if action == 0 else 0 
  dlogps.append(y - aprob)

  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward)

  if done: # an episode finished
    episode_number += 1

    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] 

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'game finnished. episode:%d, total steps %d, total reward %f. running mean: %f' % (episode_number-1,step_number-1, reward_sum, running_reward)
    reward_sum = 0
    x = env.reset()
    x_prev = None
    step_number=0
