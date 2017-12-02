""" Trains an Actor with CACLA learning through a Critic. Uses OpenAI Gym. """
# Two neural networks are used 
# Actor  is NN1, Psi parameter
# Critic is NN2, Theta parameter

import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
# decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
# resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 600 * 400 # input dimensionality: 80x80 grid
# if resume:
#   model = pickle.load(open('save.p', 'rb'))
# else:
#   model = {}
#   model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
#   model['W2'] = np.random.randn(H) / np.sqrt(H)
model = {}
model['Psi1']   = np.random.randn(H,D) / np.sqrt(D)
model['Psi2']   = np.random.randn(H) / np.sqrt(H)
model['Theta1'] = np.random.randn(H,D) / np.sqrt(D)
model['Theta2'] = np.random.randn(H) / np.sqrt(H)

xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

  
# grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
# rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def actor_forward(x):
  h1 = np.dot(model['Psi1'], x)
  h1[h1<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['Psi2'], h1)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def actor_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['Psi2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'Psi1':dW1, 'Psi2':dW2}

      def critic_forward(x,rwd):
        h2 = np.dot(model['Psi1'], x)
        h1[h1<0] = 0 # ReLU nonlinearity
        logp = np.dot(model['Psi2'], h1)
        p = sigmoid(logp)
        return delta_t, h2 # return delta_t and hidden state

      def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
          if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
          running_add = running_add * gamma + r[t]
          discounted_r[t] = running_add
        return discounted_r

      def critic_backward(eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['Psi2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'Psi1':dW1, 'Psi2':dW2}

env = gym.make("LunarLander-v2")
observation = env.reset()



while True:
  if render: env.render()


  # forward the Actor network and sample an action from the returned probability
  aprob, h1 = actor_forward(observation)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!


  # # record various intermediates (needed later for backprop)
  # xs.append(x) # observation
  # hs.append(h) # hidden state
  # y = 1 if action == 2 else 0 # a "fake label"
  # dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)


  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  delta_t, h2 = critic_forward(x,reward)
      reward_sum += reward
      # record reward (has to be done after we call step() to get reward for previous action)
      drs.append(reward) 


  if done: # an episode finished
    episode_number += 1

    # # stack together all inputs, hidden states, action gradients, and rewards for this episode
    # epx = np.vstack(xs)
    # eph = np.vstack(hs)
    # epdlogp = np.vstack(dlogps)
    # epr = np.vstack(drs)
    # xs,hs,dlogps,drs = [],[],[],[] # reset array memory

      # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
      # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    grad = policy_backward(eph, epdlogp)


    # epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    # grad = policy_backward(eph, epdlogp)
    # for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # # perform rmsprop parameter update every batch_size episodes
    # if episode_number % batch_size == 0:
    #   for k,v in model.iteritems():
    #     g = grad_buffer[k] # gradient
    #     rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
    #     model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
    #     grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # # boring book-keeping
    # running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    # print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    # if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    # reward_sum = 0
    # observation = env.reset() # reset env
    # prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
