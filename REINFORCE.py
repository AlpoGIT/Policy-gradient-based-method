import numpy as np
import torch, torch.nn as nn
import gym
from collections import deque
import matplotlib.pyplot as plt
import time

env_name = 'CartPole-v0'
env = gym.make(env_name)

# Environment
print(env_name)
print('State space:\t' + str(env.observation_space))
print('Action space:\t' + str(env.action_space))
state_dim = 4
action_dim = 2

# function approximator
# map state to prob(action|state)
class model(nn.Module):
    def __init__(self, state_dim, action_dim, fc1 = 4):
        super(model, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim = 0)
        return x

    def act(self, state):
        """Summary or Description of the Function

        Parameters:
        state: numpy state of agent

        Returns:
        action, log prob

        """

        # create distribution
        probs = self.forward(torch.from_numpy(state).type(torch.FloatTensor))
        d = torch.distributions.Categorical(probs)

        # sample from distriution
        action = d.sample()
       
        return int(action.item()), torch.log(probs[action])

# hyperparameters & utils
scores = []
average = deque(maxlen = 100)
score = 0
policy = model(state_dim, action_dim)
gamma = 0.99
alpha = 0.01
budget = 6000
optim = torch.optim.Adam(policy.parameters(), lr = alpha)

for i in range(1, budget + 1 ):
    score = 0
    rewards = []
    log_probs = []
    # generate episode
    state = env.reset()
    action, log_prob = policy.act(state)
    
    while True:
        action, log_prob = policy.act(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        score += reward
        state = next_state
        
        if done == True:
            break
            
    # loss
    gammas = np.array([gamma**t for t in range(0,len(rewards))])
    G = np.multiply(gammas, rewards)
    G = np.array([ np.sum(G[t:]*gamma**t) for t in range(0,len(rewards)) ])
    loss = -np.multiply(np.array(log_probs),G).sum()
    optim.zero_grad()
    loss.backward()
    optim.step()

    # scores
    scores.append(score)
    average.append(score)
    av_score = np.mean(average)
    if i%100 == 0:
        print('\r{}/{}\t average score: {:.2f}'.format(i,budget, av_score), end ='')
    if av_score >= 195:
        print("\nsolved in {} steps !".format(i))
        break

# plot result
average = []
scores_deque = deque(maxlen=100)

for x in scores:
    scores_deque.append(x)
    average.append(np.mean(scores_deque))
goal = [195 for x in range(len(average))]

plt.plot(scores, 'b-', label = 'score')
plt.plot(average, 'r-', label = 'average score')
plt.plot(goal, 'k--', label = 'goal')
plt.xlabel('trajectory #')
plt.ylabel('score')
plt.legend()
plt.show()


## see trained agent
#for i in range(100):
#    state = env.reset()
#    t = 0
#    while True:
#        action, _ = policy.act(state)
#        next_state, _, done, _ = env.step(action)
#        state = next_state
#        env.render()
#        time.sleep(0.1)
#        t += 1
#        if done == True:
#            print("Episode finished after {} timesteps".format(t+1))
#            time.sleep(2)
#            break
