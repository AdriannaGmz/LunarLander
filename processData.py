import gym
import numpy as np
import matplotlib.pyplot as plt
import sys

print(len(sys.argv))

if len(sys.argv) == 2:
	filepath = str(sys.argv[1])
	print(filepath)
else:
	print("Pls specify filename/path")
	sys.exit()

f = open(filepath, 'r')
lines = f.readlines()
line_no = len(lines)
ep_vec = []
step_vec = []
rew_vec = []
mean_vec = []

for k in range(line_no - 1):
	line = lines[k]
	line = line.replace("game finnished. episode:", "")
	line = line.replace(", total steps", "")
	line = line.replace(", total reward", "")
	line = line.replace(". running mean:", "")
	episode, total_steps, total_reward, running_mean = line.split(" ")
	ep_vec.append(episode)
	step_vec.append(total_steps)
	rew_vec.append(total_reward)
	mean_vec.append(running_mean)
	line = f.readline()

plt.figure(1)
plt.plot(ep_vec, step_vec)
plt.xlabel('Number of episodes')
plt.ylabel('Total steps in episode')

plt.figure(2)
plt.plot(ep_vec, rew_vec)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward per episode')

plt.figure(3)
plt.plot(ep_vec, mean_vec)
plt.xlabel('Number of episodes')
plt.ylabel('Running mean')

plt.show()