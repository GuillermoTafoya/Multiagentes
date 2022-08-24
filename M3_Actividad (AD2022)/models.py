from mesa import Model
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
        cell_content, x, y = cell        
        for agent in cell_content:
            #print(agent, x, y)
            if isinstance(agent, Car):
                if agent.colour == 'white':
                    grid[x][y] = 6
                elif agent.colour == 'blue':
                    grid[x][y] = 7
                elif agent.colour == 'purple':
                    grid[x][y] = 8
                else: # black
                    grid[x][y] = 9
            
            elif isinstance(agent, TrafficLight):
                if agent.state == True: # Green
                    grid[x][y] = 2
                else: # red
                    grid[x][y] = 1

            elif isinstance(agent, Road):
                
                if agent.colour == "brown":
                    grid[x][y] = 3
                elif agent.colour == 'olive':
                    grid[x][y] = 4
                else: # dark green
                    grid[x][y] = 5
            
            else: # Street
                grid[x][y] = 0

    return grid


class Board(Model):
    def __init__(self, width, height, N, seed=None):
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid})
        random.seed(seed if seed is not None else time.time())
        self.N = N
        self.create_agents()
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def create_agents(self):
        CarsId = 0
        TrafficLightsId = self.N+1
        # Create roads
        # Makes a crossroad
        for (content,x,y) in self.grid.coord_iter():
            if x < self.width // 3 or x > self.width * 2 // 3:
                if y < self.height // 3 or y > self.height * 2 // 3:
                #print("1. Added road at", x, y)
                    road = Road((x,y), self, colour = "olive")
                    self.grid.place_agent(road, (x, y))
                    self.schedule.add(road)

            # Create cars on the sides of the crossroad             
            if CarsId < self.N:
                # Cars going up
                if x == 0 and abs(y - self.height // 2) < 2:
                    #print("3. Added car at", x, y)
                    car = Car(CarsId, self, direction = "right", colour = "white")
                    self.grid.place_agent(car, (x, y))
                    self.schedule.add(car)
                    CarsId += 1
                # Cars going down
                elif abs(x - self.width // 2) < 2 and y == 0:
                    #print("4. Added car at", x, y)
                    car = Car(CarsId, self, direction = "up", colour = "blue")
                    self.grid.place_agent(car, (x, y))
                    self.schedule.add(car)
                    CarsId += 1
        # Create traffic lights
        # Only two traffic lights, one from up to down and one from right to left
        trafficLight = TrafficLight(self.N + 1, self, direction = "left")
        self.grid.place_agent(trafficLight, (self.width // 3, self.height // 3 * 2))
        self.schedule.add(trafficLight)
        
        trafficLight = TrafficLight(self.N + 2, self, direction = "down")
        self.grid.place_agent(trafficLight, (self.width // 3 * 2, self.height // 3))
        self.schedule.add(trafficLight)

if __name__ == '__main__':
    # Test if the model works
    board = Board(10, 10, 10)
    for i in range(10):
        board.step()
        all = board.datacollector.get_model_vars_dataframe()
        print(all.iloc[i]['Grid'])