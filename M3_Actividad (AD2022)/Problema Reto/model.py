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

def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))

    #Por todas las celdas del grid
    for cell in model.grid.coord_iter():
        agent, x, y = cell

        if isinstance(agent, Car):
            if agent.colour == 'white':
                grid[x][y] = 6
            elif agent.colour == 'blue':
                grid[x][y] = 7
            elif agent.colour == 'purple':
                grid[x][y] = 8
            else: # black
                grid[x][y] = 9

        elif isinstance(agent, Road):
            if agent.colour == 'brown':
                grid[x][y] = 3
            elif agent.colour == 'olive':
                grid[x][y] = 4
            else: # dark green
                grid[x][y] = 5

        elif isinstance(agent, TrafficLight):
            if agent.colour == 'green':
                grid[x][y] = 2
            else: # red
                grid[x][y] = 1

        else: # Street
            grid[x][y] = 0

    return grid