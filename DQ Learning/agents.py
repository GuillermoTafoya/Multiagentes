#Aiuda

from re import S
from mesa import Agent

class Car(Agent):
    DIRECTIONS = ['right', 'down', 'left', 'up']

    def __init__(self, unique_id, model, colour=None, direction=None):
        super().__init__(unique_id, model)
        self.colour = colour
        self.dx = 0
        self.dy = 0
        self._direction = self.random.choice(self.DIRECTIONS) if not direction else direction
        self.direction = self._direction
        self.alive = True
        self.successful_trip = False
        self.stopped = False
        self.next_pos = unique_id

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = direction
        if self._direction == 'up':
            self.dx, self.dy = -1, 0
            return
        if self._direction == 'down':
            self.dx, self.dy = 1, 0
            return
        if self._direction == 'right':
            self.dx, self.dy = 0, 1
            return
        if self._direction == 'left':
            self.dx, self.dy = 0, -1
            return
        raise(ValueError('Invalid direction'))

    def opositeDirections(self, direction1, direction2):
        if direction1 == 'up' and direction2 == 'down':
            return True
        if direction1 == 'down' and direction2 == 'up':
            return True
        if direction1 == 'right' and direction2 == 'left':
            return True
        if direction1 == 'left' and direction2 == 'right':
            return True
        return False

    def step(self):
        """
        Defines how the model interacts within its environment.
        """
        if not self.alive:
            return
        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=max(self.model.width, self.model.height))
        for neighbour in neighbours:

            if isinstance(neighbour, TrafficLight):
                if neighbour.state == True and self.opositeDirections(neighbour.direction, self.direction):
                    # stop
                    if (self.direction == 'down') and neighbour.pos[0] - self.pos[0] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                    if (self.direction == 'up') and self.pos[0] - neighbour.pos[0] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                    if (self.direction == 'right') and neighbour.pos[1] - self.pos[1] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                    if (self.direction == 'left') and self.pos[1] - neighbour.pos[1] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
        self.stopped = False
        self.next_pos = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        
        

        
        
    def advance(self):
        if self.stopped:
            self.next_pos = self.pos
            return
        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=max(self.model.width, self.model.height))
        for neighbour in neighbours:
            # Try stopping if there is another car in the way
            if isinstance(neighbour, Car):
                if (self.direction == neighbour.direction == 'down') and neighbour.pos[0] - self.pos[0] == 1 and neighbour.stopped:
                    if neighbour.pos[1] == self.pos[1]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                elif (self.direction == neighbour.direction == 'up') and self.pos[0] - neighbour.pos[0] == 1 and neighbour.stopped:
                    if neighbour.pos[1] == self.pos[1]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                elif (self.direction == neighbour.direction == 'right') and neighbour.pos[1] - self.pos[1] == 1 and neighbour.stopped:
                    if neighbour.pos[0] == self.pos[0]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                elif (self.direction == neighbour.direction == 'left') and self.pos[1] - neighbour.pos[1] == 1 and neighbour.stopped:
                    if neighbour.pos[0] == self.pos[0]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                # Check collision with cars
                if self.next_pos == neighbour.next_pos and neighbour is not self and self.direction != neighbour.direction and neighbour.stopped == self.stopped == False:
                    self.alive = False
                    neighbour.alive = False
                    return
        # Check if the car has reached the goal
        if self.direction == 'down' and self.next_pos[0] == self.model.width - 1:
            self.successful_trip = True
        elif self.direction == 'up' and self.next_pos[0] == 0:
            self.successful_trip = True
        elif self.direction == 'right' and self.next_pos[1] == self.model.height - 1:
            self.successful_trip = True
        elif self.direction == 'left' and self.next_pos[1] == 0:
            self.successful_trip = True
        self.model.grid.move_agent(self, self.next_pos)

class TrafficLight(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, state = True, timeToChange = 10, direction = None, delay = 0):
        super().__init__(unique_id, model)
        self.state = state
        self.timeToChange = timeToChange
        self._timeToChange = timeToChange
        self.direction = direction
        self._delay = delay
        self.delay = 0

    def step(self):
        if self.state and self.delay < self._delay:
            self.delay += 1
            return
        if self.state and self.delay >= self._delay:
            self._delay = 0
            self.timeToChange -= 1
            if self.timeToChange == 0:
                self.state = False
                self.timeToChange = self._timeToChange
        else:
            self.timeToChange -= 1
            if self.timeToChange == 0:
                self.state = True
                self.timeToChange = self._timeToChange

    #def advance(self) -> None:
        
        

class Road(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, colour = None):
        super().__init__(unique_id, model)
        self.colour = colour if colour else "olive"
    def step(self):
        pass

"""import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import deque
import random
import math"""

"""class TrafficLightIA(Agent):
    """
    #Agent to be trained using DQ learning.
"""
    class Linear_QNet(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_size)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

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
    
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def getMove(self, state, epsilon = 1000):
        possible_moves = []
        for cell in range(len(state[0])):
            if state[0][cell] == 0:
                possible_moves.append(cell)
        #print("Possible Moves:",possible_moves)
        if not possible_moves:
            return (-1,1)
        if (np.random.random() <= epsilon):
            #print("Random Play")
 
            n = random.choice(possible_moves)
            return (n//3,n%3) 

                
        else:
            with torch.no_grad():
                #print("DQ Play")
                prediction = torch.topk(self.DQN(state),9) ### !!!.cpu()
                preferedMoves = prediction[1].numpy()

                for n in preferedMoves[0]:
                    if int(n) in possible_moves:
                        return (n//3,n%3) if possible_moves else (-1,1)
  

    def remember(self, state, action, reward, next_state, done, e): ###
        reward = torch.tensor(reward)#.to(self.device)
        self.memory.append((state, action, reward, next_state, done, e))
    
    def replay(self, batch_size):
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done, e in minibatch: ###
            y = self.DQN(state)
            y_target = y.clone().detach()
            with torch.no_grad():
                #print("Trying:",self.DQN(next_state)[0])
                #print("Debug:",self.getMove(next_state, self.get_epsilon(e)))
                r,c = self.getMove(next_state, self.get_epsilon(e))
                ac = r*3+c
                #print("Action:",ac)
                y_target[0][action] = reward if (done or ac<0) else reward + self.gamma * self.DQN(next_state)[0][ac] ###
                
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
    
"""