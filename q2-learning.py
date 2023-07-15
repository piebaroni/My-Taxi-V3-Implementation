import numpy as np
import random
from matplotlib import pyplot as plt
from mytaxy import TaxiEnv2
from statistics import mean

env = TaxiEnv2()

Q = np.zeros([env.observation_space.n,env.action_space.n])
gamma = 0.9  #Discount factor
alpha = 0.2   #Learning rate
epsilon = 0.1 #epsilon-greedy
cum_reward = 0
reward_list = []
actions_list = []
for episode in range(10000):
    done = False
    episode_reward = 0
    state = env.reset()[0]
    actions = 0
    while not done:
        actions += 1
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #Explore state space
        else:
            action = np.argmax(Q[state]) #Exploit learned values
        next_state, reward, done, info = env.step(action) #Invoke Gym
        next_max = np.max(Q[next_state])
        old_value = Q[state, action]
        #Q-Learning update rule

        #SARSA Update Rule
        new_value = old_value + alpha * (reward + gamma * Q[next_state,np.argmax(Q[next_state])] - old_value)
        Q[state, action] = new_value
        episode_reward += reward
        state = next_state
        #if actions > 100000:
            #done = True
    cum_reward+=episode_reward
    reward_list.append(episode_reward)
    actions_list.append(actions)
    if episode % 100 == 0:
        print("Episode {} Total Reward: {}".format(episode, episode_reward))

plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward for each episode for my TaxiV3 implementation')
#plt.show()
plt.savefig(r'C:\Users\rocci\OneDrive\Desktop\Università\ATCS\Reinforcement Learning\rewards_episode.png')

plt.clf()

plt.plot(actions_list)
plt.xlabel('Episode')
plt.ylabel('Movements')
plt.title('Number of steps for each episode for my TaxiV3 implementation')
plt.savefig(r'C:\Users\rocci\OneDrive\Desktop\Università\ATCS\Reinforcement Learning\movements_episode.png')

print('###########################################')
print('Max Reward: ' + str(max(reward_list)) + ', Min Reward: ' + str(min(reward_list)) + ', Average Reward: ' + str(mean(reward_list)))
print('Max Steps: ' + str(max(actions_list)) + ', Min Steps: ' + str(min(actions_list)) + ', Average Steps: ' + str(mean(actions_list)))
print('###########################################')
