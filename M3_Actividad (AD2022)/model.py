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

from agents import Car, TrafficLight, Road

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
            
            if agent.colour == "brown":
                grid[x][y] = 3
            elif agent.colour == 'olive':
                grid[x][y] = 4
            else: # dark green
                grid[x][y] = 5

        elif isinstance(agent, TrafficLight):
            if agent.state == True: # Green
                grid[x][y] = 2
            else: # red
                grid[x][y] = 1

        else: # Street
            grid[x][y] = 0

    return grid

    class Board(model):
        def __init__(self, width, height, N, seed=None):
            self.width = width
            self.height = height
            self.grid = MultiGrid(width, height, torus=True)
            self.schedule = SimultaneousActivation(self)
            self.running = True
            self.datacollector = DataCollector(
                model_reporters={"Grid": get_grid},
                agent_reporters={"x": "x", "y": "y", "colour": "colour", "direction": "direction"}
            )
            self.seed = seed
            self.N = N
            self.create_agents()
            self.datacollector.collect(self)
        def create_agents(self):
            # Create roads
            # Makes a crossroad
            for i in range(self.width):
                for j in range(self.height):
                    if i == self.width // 2 or j == self.height // 2:
                        road = Road((i, j), self, 'brown')
                        self.grid.place_agent(road, (i, j))
                        self.schedule.add(road)
                    """
                    else:
                        road = Road((i, j), self, 'olive')
                        self.grid.place_agent(road, (i, j))
                        self.schedule.add(road)
                    """
            # Create cars on the sides of the crossroad 
            placed = 0
            for i in range(self.width):
                for j in range(self.height):
                    if i == self.width // 2 and j == self.height // 2:
                        continue
                    if placed == self.N:
                        break
                    # Cars going up
                    if i == self.width // 2 or j == self.height // 2:
                        car = Car((i, j), self, 'white', 'up')
                        self.grid.place_agent(car, (i, j))
                        self.schedule.add(car)
                    # Cars going from left to right
                    elif i < self.width // 2 and j < self.height // 2:
                        car = Car((i, j), self, 'blue', 'right')
                        self.grid.place_agent(car, (i, j))
                        self.schedule.add(car)
            # Create traffic lights
            # Only two traffic lights, one from up to down and one from right to left
            for i in range(self.width):
                for j in range(self.height):
                    if i == self.width // 2 and j == self.height // 2:
                        traffic_light = TrafficLight((i, j), self, True, 10, "up")
                        self.grid.place_agent(traffic_light, (i, j))
                        self.schedule.add(traffic_light)
                    elif i == self.width // 2 and j == self.height // 2 - 1:
                        traffic_light = TrafficLight((i, j), self, True, 10, "up")
                        self.grid.place_agent(traffic_light, (i, j))
                        self.schedule.add(traffic_light)