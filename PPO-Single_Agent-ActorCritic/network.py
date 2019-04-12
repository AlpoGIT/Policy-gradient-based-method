import numpy as np
import torch, torch.nn as nn
from collections import deque, namedtuple
import random
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class actor_critic_continuous(nn.Module):
    def __init__(self, state_dim, action_dim, fc1 = 64, fc2 = 64):
        super(actor_critic_continuous, self).__init__()
        
        self.actor =  nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, action_dim),
                                    nn.Tanh()
                                    )
        self.critic = nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, 1)
                                    )
        
        
        self.sigma = nn.Parameter(torch.abs(torch.randn(action_dim)))
        #self.cov = nn.Parameter(torch.eye(action_dim))
        #self.sigma = torch.full((1,action_dim ), 0.1)

    def act(self, state):
        #x = torch.as_tensor(state, dtype = torch.float).to(device)
        mean = self.actor(state)
        #torch.nn.functional.softplus(
        #d = torch.distributions.Normal(mean, torch.nn.functional.softplus(self.sigma))
        d = torch.distributions.Normal(mean, self.sigma)
        action = d.sample()
        log_prob = d.log_prob(action)
        return action.cpu().numpy() , log_prob.sum(-1).unsqueeze(-1), d.entropy().sum(-1).unsqueeze(-1)

    def value(self, state):
        #x = torch.as_tensor(state, dtype = torch.float).to(device)
        return self.critic(state)

    def give_log_prob(self, state, action):
        mean = self.actor(state)
        #d = torch.distributions.Normal(mean, torch.nn.functional.softplus(self.sigma))
        d = torch.distributions.Normal(mean,self.sigma)
        log_prob = d.log_prob(action)
        return log_prob.sum(-1).unsqueeze(-1), d.entropy().sum(-1).unsqueeze(-1)




class actor_critic_continuous_covariant(nn.Module):
    def __init__(self, state_dim, action_dim, fc1 = 64, fc2 = 64):
        super(actor_critic_continuous_covariant, self).__init__()
        
        self.actor =  nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, action_dim),
                                    nn.Tanh()
                                    )
        self.critic = nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, 1)
                                    )
        #torch.nn.init.normal_( tensor, mean=0, std=1)
        #self.sigma = nn.Parameter(torch.ones(action_dim))
        self.cov = nn.Parameter(torch.eye(action_dim))

    def act(self, state):
        #x = torch.as_tensor(state, dtype = torch.float).to(device)
        mean = self.actor(state)
        #torch.nn.functional.softplus(
        #d = torch.distributions.Normal(mean, torch.nn.functional.softplus(self.sigma))
        d = torch.distributions.MultivariateNormal(mean, self.cov)
        action = d.sample()
        log_prob = d.log_prob(action)


        return action.cpu().numpy() , log_prob.unsqueeze(-1), d.entropy().unsqueeze(-1)

    def value(self, state):
        #x = torch.as_tensor(state, dtype = torch.float).to(device)
        return self.critic(state)

    def give_log_prob(self, state, action):
        mean = self.actor(state)
        #d = torch.distributions.Normal(mean, torch.nn.functional.softplus(self.sigma))
        d = torch.distributions.MultivariateNormal(mean, self.cov)
        log_prob = d.log_prob(action)

        return log_prob.unsqueeze(-1), d.entropy().unsqueeze(-1)





class actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, fc1 = 8, fc2 = 8):
        super(actor_network, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.nn.functional.softmax(x, dim = -1)

        return x

    def act(self, state):
        x = torch.as_tensor(state, dtype = torch.float).to(device)
        probs = self.forward(x)
        d = torch.distributions.Categorical(probs)
        action = d.sample()
        index = int(action.item())

        return index , probs[index], d.entropy()


class critic_network(nn.Module):
    def __init__(self, state_dim, output_dim=1, fc1=8, fc2=8):
        super(critic_network, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)

        return x

    def value(self, state):
        x = torch.as_tensor(state, dtype = torch.float).to(device)
        return self.forward(x)


class storage:
    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        # (state, action, reward, done, proba, value)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "proba",  "value"])
        
    
    def add(self, state, action, reward, done, proba, value):
        e = self.experience(state, action, reward, done, proba, value)
        self.memory.append(e)
    
    def sample(self, advantages, returns):
        """generate list of mini_batch"""
        batch_size = self.batch_size
        all_batch = []
        indices = np.random.permutation(np.arange(self.__len__())) 
        #indices = np.arange(self.__len__())
        batch_number = int(self.__len__() / self.batch_size)




        for k in np.arange(batch_number-1):
            experiences = [self.memory[i] for i in indices[k*batch_size : (k + 1)*batch_size]]
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            probas = torch.from_numpy(np.vstack([e.proba for e in experiences if e is not None])).float().to(device)
            values = torch.from_numpy(np.vstack([e.value for e in experiences if e is not None])).float().to(device)
            #entropies = torch.from_numpy(np.vstack([e.entropy for e in experiences if e is not None])).float().to(device)
            #entropies = torch.stack([e.entropy for e in experiences if e is not None], dim = 0).unsqueeze(1).to(device)
            batch_returns = torch.stack( [returns[i] for i in indices[k*batch_size : (k+1)*batch_size]] , dim = 0).to(device)
            batch_advantages = torch.from_numpy(np.vstack( [advantages[i] for i in indices[k*batch_size : (k+1)*batch_size]])).float().to(device)

            #advantages = torch.from_numpy(np.vstack([e.advantage for e in experiences if e is not None])).float().to(device)
            
            batch = (states, actions, rewards, dones, probas, values, batch_returns, batch_advantages)
            all_batch.append(batch)
        
       #last batch is longer
        experiences = [self.memory[i] for i in indices[(batch_number-1)*batch_size:]]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        probas = torch.from_numpy(np.vstack([e.proba for e in experiences if e is not None])).float().to(device)
        values = torch.from_numpy(np.vstack([e.value for e in experiences if e is not None])).float().to(device)
        #advantages = torch.stack([e.advantage for e in experiences if e is not None], dim = 0).unsqueeze(1).to(device)
        #entropies = torch.stack([e.entropy for e in experiences if e is not None], dim = 0).unsqueeze(1).to(device)
        batch_returns = torch.stack( [returns[i] for i in indices[(batch_number-1)*batch_size:]] , dim = 0).to(device)
        #batch_advantages = torch.stack( [advantages[i] for i in indices[(batch_number-1)*batch_size:]] , dim = 0).to(device)
        batch_advantages = torch.from_numpy(np.vstack( [advantages[i] for i in indices[(batch_number-1)*batch_size:]])).float().to(device)

        #advantages = torch.from_numpy(np.vstack([e.advantage for e in experiences if e is not None])).float().to(device)
            
        batch = (states, actions, rewards, dones, probas, values, batch_returns, batch_advantages)
        all_batch.append(batch)

        return all_batch

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)

    def clear(self):
        self.memory.clear()

