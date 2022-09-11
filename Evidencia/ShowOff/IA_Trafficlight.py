import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque
import random
import math
# To be implemented
class TrafficLightIA():
    """
    # Agent IA to be trained using DQ learning.
    """
    class Q_Net(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            # Process a 31x31x11 grid (11 channels)
            # It is resized to 1x(31x31x11)
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 256)
            # Return an array with the Q values for the 4 trafficLights(each action represents a different permutation of the traffic lights)
            # After the processing, the output should be a 1x16 array
            self.fc3 = nn.Linear(256, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def save(self, path):
            torch.save(self.state_dict(), path)

    def __init__(self, model, n_episodes=1000, n_wins_objective=100, max_env_steps=None, gamma=1.0, epsilon=1.0,
                    epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, 
                    batch_size=64, quiet=False, saveweights=True):
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
        self.criterion = torch.nn.MSELoss()
        self.saveweights=saveweights
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state = False
        # Load DQN.pth if it exists
        if not os.path.isfile("DQN.pth"):
            self.DQN = self.Q_Net(11*31*31, 16).to(self.device)
        else:
            self.DQN = self.Q_Net(11*31*31, 16).to(self.device)
            self.DQN.load_state_dict(torch.load("DQN.pth"))

        self.opt = torch.optim.Adam(self.DQN.parameters(), lr=0.01)
    
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def getMoves(self, state, epsilon = 1000):
        """
        Return the Q values for each action
        """
        if (np.random.random() <= epsilon):
            return np.random.randint(0, 16),True
        else:
            with torch.no_grad():
                return torch.argmax(self.DQN(state)).cpu().numpy(),False

    def remember(self, state, action, reward, next_state, done, e): ###
        reward = torch.tensor(reward).to(self.device)
        self.memory.append((state, action, reward, next_state, done, e))
    
    def replay(self, batch_size):
        """
        Replay the memory to train the DQN
        """
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done, e in minibatch:
            y = self.DQN(state)
            y_target = y.clone().detach()
            with torch.no_grad():
                y_target[0][action] = reward if done else reward + self.gamma * torch.max(self.DQN(next_state)[0])
            y_batch.append(y[0])
            y_target_batch.append(y_target[0])
        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)
        self.opt.zero_grad()
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
        maxSteps = 1000
        for e in range(self.n_episodes):
            state = self.env.reset()
            state = torch.tensor(state).to(self.device)
            # We reshape the state to be a 1x(11x31x31) tensor
            state = state.view(1, 11*31*31)
            print("-"*50)
            print("Episode: ", e)
            print("-"*50)
            state = state.float()
            score = 0
            done = False
            steps = 0
            randomActions = 0
            actions = 0
            totalCrashes = 0
            totalTimeStuck = 0
            while not done:
                action,wasRandom = self.getMoves(state, self.get_epsilon(e))
                if wasRandom:
                    randomActions+=1
                actions+=1
                reward = 0
                # Give time so the cars can move
                for _ in range(3): 
                    next_state, r, done, _ = self.env.step(action)
                    reward+=r
                    totalCrashes+=self.env.crashes
                    totalTimeStuck+=self.env.time_stuck
                    self.env.time_stuck = 0
                    self.env.crashes = 0
                next_state = torch.tensor(next_state).to(self.device)
                next_state = next_state.view(1,11*31*31)
                next_state = next_state.float()
                if steps % 20 == 0:
                    print(f'[{e}] [{steps}]',"Action: ", action, "| Reward: ", reward, "| Was random action:", wasRandom)
                    print("Successful Trips:", self.env.successful_trips, "| Crashes:", totalCrashes, "| Stuck Time:", totalTimeStuck)
                    print("Total Random Actions:", randomActions, "| Total Actions:", actions)
                self.remember(state, action, reward, next_state, done, e)
                state = next_state
                score += reward
                steps += 1
                if steps > maxSteps:
                    break
            scores.append(score)
            mean_score = np.mean(scores)
            if mean_score >= self.n_wins_objective and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                if self.saveweights: torch.save(self.DQN.state_dict(), 'DQN.pth')
                break
            if e % 5 == 0 and not self.quiet:
                print('[Episode {}] - Mean score over last 5 episodes was {}.'.format(e, mean_score))
            self.replay(self.batch_size)
            self.Q_Net.save(self.DQN, 'DQN.pth')
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e
