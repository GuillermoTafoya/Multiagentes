from turtle import forward
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque
import random
import math
from models import Board
from agents import Agent

# To be implemented
class TrafficLightIA(Agent):
    """
    #Agent to be trained using DQ learning.
    """
    class Q_Net(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            # Process a 31x31x11 grid (11 channels)
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            # Return a 1x4 array with the Q values for the 4 trafficLights(the action is switching, for the 4 traffic lights)
            # After the convolution, the output should be a 1x4
            self.fc3 = nn.Linear(128, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def save(self, path):
            torch.save(self.state_dict(), path)

    def __init__(self, unique_id, model, n_episodes=1000, n_wins_objective=195, max_env_steps=None, gamma=1.0, epsilon=1.0,
                    epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, 
                    batch_size=64, quiet=False, saveweights=True):
        super().__init__(unique_id, model)
        self.memory = deque(maxlen=100000)
        
        self.env = model
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_wins_objective = n_wins_objective
        self.batch_size = batch_size
        self.quiet = quiet
        self.saveweights=saveweights
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        
        self.state = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DQN = self.Q_Net(11*31*31, 4).to(self.device)
    
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def getMoves(self, state, epsilon = 1000):
        """
        Return the Q values for each action (on or off for the 4 traffic lights)
        """
        if (np.random.random() <= epsilon):
            # Select a random action
            return np.random.randint(0, 2, size=4)
        else:
            with torch.no_grad():
                prediction = self.DQN(state)
                # We return all the Q values, and each agent will choose if it switches or not
                # depending on its corresponding Q value
                # Return an array of 0s and 1s, 1 if the agent switches, 0 if it doesn't
                return prediction.cpu().numpy()

    def remember(self, state, action, reward, next_state, done, e): ###
        reward = torch.tensor(reward).to(self.device)
        self.memory.append((state, action, reward, next_state, done, e))
    
    def replay(self, batch_size):
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done, e in minibatch: ###
            y = self.DQN(state)
            y_target = y.clone().detach()
            with torch.no_grad():
                switches = self.getMove(next_state, self.get_epsilon(e))
                # We get the Q values for the next state
                y_target_next = self.DQN(next_state)
                # We get the Q value for the action that was taken
                Q_next = y_target_next[switches]
                # We get the Q value for the action that was taken
                Q_next = Q_next.max(1)[0].view(1, 1)
                # If the episode is done, we don't add the Q value of the next state
                if done:
                    y_target[0][action] = reward
                else:
                    y_target[0][action] = reward + self.gamma * Q_next
            y_batch.append(y)
            y_target_batch.append(y_target)
        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)
        loss = F.mse_loss(y_batch, y_target_batch)
        self.DQN.zero_grad()
        loss = self.criterion(y_batch, y_target_batch)
        loss.backward()
        self.opt.step()        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def train(self):
        """
        Train the agent using DQ learning.
        """
        scores = deque(maxlen=100)
        for e in range(self.n_episodes):
            state = self.env.reset()
            state = torch.tensor(state).to(self.device)
            state = torch.reshape(state, (1, 11, 31, 31))
            score = 0
            done = False
            while not done:
                switches = self.getMoves(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(switches)
                next_state = torch.tensor(next_state).to(self.device)
                next_state = torch.reshape(next_state, (1, 11, 31, 31))
                self.remember(state, switches, reward, next_state, done, e)
                state = next_state
                score += reward
            scores.append(score)
            mean_score = np.mean(scores)
            if mean_score >= self.n_wins_objective and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                if self.saveweights: torch.save(self.DQN.state_dict(), 'DQN.pth')
                break
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
            self.replay(self.batch_size)
            self.Q_Net.save(self.DQN, 'DQN.pth')
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e
