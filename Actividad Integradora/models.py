from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2 ** 128
import numpy as np
import pandas as pd
import time
import datetime
import random
from agents import *

def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell        
        for agent in cell_content:
            if isinstance(agent, Box):
                grid[x][y] = 1
                continue
            if isinstance(agent, Stack):
                # Depending on the capacity of the stack, the color will change
                if agent.capacity == 0:
                    grid[x][y] = 2
                elif agent.capacity == 1:
                    grid[x][y] = 3
                elif agent.capacity == 2:
                    grid[x][y] = 4
                elif agent.capacity == 3:
                    grid[x][y] = 5
                elif agent.capacity == 4:
                    grid[x][y] = 6
                elif agent.capacity == 5:
                    grid[x][y] = 7
                break

        for agent in cell_content:
            if isinstance(agent, Robot):
                if agent.cargo:
                    grid[x][y] = 8
                else:
                    grid[x][y] = 9
                break
    return grid

class Garage(Model):
    def __init__(self, R, B, width, height):
        self._num_robots = R
        self._num_boxes = B
        self.grid = MultiGrid(width, height, torus = False)
        self.schedule = SimultaneousActivation(self)
        self.width = width
        self.height = height
        self.init()
        self.datacollector = DataCollector(
            model_reporters = {"Grid": get_grid}
        )
    def init(self):
        # Place stacks only in the upper row
        for i in range(0, self.width):
            stack = Stack((i, 0), self, (i, 0))
            self.schedule.add(stack)
            self.grid.place_agent(stack,(i,0))

        # Create robots
        for i in range(self._num_robots):
            robot = Robot(i, self)
            self.grid.place_agent(robot, robot.pos)
            self.schedule.add(robot)
            
        # Create boxes
        for i in range(self._num_robots,self._num_robots+self._num_boxes):
            box = Box(i, self)
            
            self.grid.place_agent(box, box.pos)
            self.schedule.add(box)
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

