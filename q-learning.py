import gym
import numpy as np
import random
from matplotlib import pyplot as plt
from statistics import mean

#Environment Setup
env = gym.make("Taxi-v3")

env.reset()
#env.render()

#Q[state, action] table implementation
Q = np.zeros([env.observation_space.n,env.action_space.n])
gamma = 0.9  #Discount factor
alpha = 0.2   #Learning rate
epsilon = 0.1 #epsilon-greedy
cum_reward = 0
reward_list = []
actions_list = []
for episode in range(10000):
    terminated = False
    actions = 0
    episode_reward = 0
    state = env.reset()[0]
    while not terminated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #Explore state space
        else:
            action = np.argmax(Q[state]) #Exploit learned values
        next_state, reward, terminated, done, info = env.step(action) #Invoke Gym
        next_max = np.max(Q[next_state])
        old_value = Q[state, action]
        #print(env.step(action))
        #Q-Learning update rule

        #SARSA Update Rule
        new_value = old_value + alpha * (reward + gamma * Q[next_state,np.argmax(Q[next_state])] - old_value)
        Q[state, action] = new_value
        episode_reward += reward
        state = next_state
        actions += 1
    cum_reward+=episode_reward
    reward_list.append(episode_reward)
    actions_list.append(actions)
    if episode % 100 == 0:
        print("Episode {} Total Reward: {}".format(episode, episode_reward))

plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward for each episode for TaxiV3')
plt.savefig(r'C:\Users\rocci\OneDrive\Desktop\Università\ATCS\Reinforcement Learning\rewards_episode_original.png')

plt.clf()

plt.plot(actions_list)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Number of steps for each episode for TaxiV3')
plt.savefig(r'C:\Users\rocci\OneDrive\Desktop\Università\ATCS\Reinforcement Learning\movements_episode_original.png')

print('###########################################')
print('Max Reward: ' + str(max(reward_list)) + ', Min Reward: ' + str(min(reward_list)) + ', Average Reward: ' + str(mean(reward_list)))
print('Max Steps: ' + str(max(actions_list)) + ', Min Steps: ' + str(min(actions_list)) + ', Average Steps: ' + str(mean(actions_list)))
print('###########################################')