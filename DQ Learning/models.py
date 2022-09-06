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
import random
from agents import Car, TrafficLight, Road

def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    json = "{"
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell        
        for agent in cell_content:
            # Save relevant agents to json
            if isinstance(agent, Car):
                json += f'"{agent.unique_id}":{{"x":{x},"y":{y},"speed":{agent.dx,agent.dy},"direction":"{agent.direction}"}},'
                if agent.direction == 'up':
                    grid[x][y] = 4
                elif agent.direction == 'down':
                    grid[x][y] = 5
                elif agent.direction == 'left':
                    grid[x][y] = 6
                elif agent.direction == 'right':
                    grid[x][y] = 7
            elif isinstance(agent, TrafficLight):
                if agent.state == True:
                    grid[x][y] = 1
                else:
                    grid[x][y] = 2
            elif isinstance(agent, Road):
                grid[x][y] = 3
        if not cell_content:
            if model.grid.width//2 == x :
                if (y+1)%2 == 0:
                    continue
                if y < model.grid.height//3 or y > 2*model.grid.height//3:
                    grid[x][y] = 8
            elif model.grid.height//2 == y :
                if (x+1)%2 == 0:
                    continue
                if x < model.grid.width//3 or x > 2*model.grid.width//3:
                    grid[x][y] = 8
    json = json[:-1] + "}"
    # Append json to file
    with open("data.txt", "a") as file:
        file.write(json + "\n")
    return grid

def get_grid_one_hot(model):
    grid = np.zeros((model.grid.width, model.grid.height, 5))
    #Por todas las celdas del grid
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell        
        for agent in cell_content:
            # Save relevant agents to json
            if isinstance(agent, Car):
                if agent.colour == 'white':
                    grid[x][y][2] = 1
                elif agent.colour == 'blue':
                    grid[x][y][3] = 1
            elif isinstance(agent, TrafficLight):
                if agent.state == True:
                    grid[x][y][0] = 1
                else:
                    grid[x][y][1] = 1
            elif isinstance(agent, Road):
                grid[x][y][4] = 1
            else: # Street
                pass
    return grid
class Board(Model):
    def __init__(self, height, width, seed=None, spawn_rate = 1, max_spawn_batch = 1, one_hot = False):
        self.width = height
        self.height = width
        self.grid = MultiGrid(height, width, torus=False)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid_one_hot if one_hot else get_grid})
        random.seed(seed if seed is not None else time.time())
        self.carID = 4
        self.spawn_rate = spawn_rate
        self.crashes = 0
        self.successful_trips = 0
        self.max_spawn_batch = max_spawn_batch
        self.create_agents()
    def step(self):
        if self.schedule.steps % self.spawn_rate == 0:
            for _ in range(random.randint(1, self.max_spawn_batch)):
                self.spawn_random_car()
                
        self.datacollector.collect(self)
        self.schedule.step()
        for agent in self.schedule.agents:
            if isinstance(agent, Car):
                if not agent.alive:
                    self.crashes += 0.5 # 0.5 because it counts both cars
                    self.schedule.remove(agent)
                    self.grid.remove_agent(agent)
                    del agent
                    continue
                if agent.successful_trip:
                    self.successful_trips += 1
                    self.schedule.remove(agent)
                    self.grid.remove_agent(agent)
                    del agent
                    continue
    def spawn_random_car(self):
        direction = random.choice(['left', 'right', 'up', 'down'])
        # Leave separation between lanes
        if direction == 'left':
            y = self.height - 1
            x = random.randint(self.width // 2 + 1, self.width * 2 // 3)
        elif direction == 'right':
            y = 0
            x = random.randint(self.width // 3, self.width // 2 - 1)
        elif direction == 'down':
            y = random.randint(self.height // 2 + 1, self.height * 2 // 3)
            x = 0
        elif direction == 'up':
            y = random.randint(self.height // 3, self.height // 2 - 1)
            x = self.width - 1
        car = Car(self.carID, self, direction = direction)
        self.carID += 1
        # Check if there is a car in the spawn position
        if self.grid.is_cell_empty((x, y)):
            self.grid.place_agent(car, (x, y))
            self.schedule.add(car)
        else:
            del car
    def create_agents(self):
        # Create roads
        # Makes a crossroad
        for (_,x,y) in self.grid.coord_iter():
            if x < self.width // 3 or x > self.width // 3 * 2:
                if y < self.height // 3 or y > self.height // 3 * 2:
                    road = Road((x,y), self, colour = "olive")
                    self.grid.place_agent(road, (x, y))
                    self.schedule.add(road)
        # Create traffic lights
        # Four traffic lights, on the corners of the crossroad
        trafficLight = TrafficLight(0, self, state=False, timeToChange=self.height, direction = "up", delay = self.height)
        self.grid.place_agent(trafficLight, (self.width // 3 - 1, self.height // 3 * 2))
        self.schedule.add(trafficLight)
        trafficLight = TrafficLight(1, self, state=False, timeToChange=self.height, direction = "down", delay = self.width)
        self.grid.place_agent(trafficLight, (self.width // 3 * 2 + 1, self.height // 3))
        self.schedule.add(trafficLight)
        trafficLight = TrafficLight(2, self, state=True, timeToChange=self.width, direction = "right", delay = self.width)
        self.grid.place_agent(trafficLight, (self.width // 3 * 2, self.height // 3 * 2 + 1))
        self.schedule.add(trafficLight)
        trafficLight = TrafficLight(3, self, state=True, timeToChange=self.width, direction = "left", delay = self.width)
        self.grid.place_agent(trafficLight, (self.width // 3, self.height // 3 - 1))
        self.schedule.add(trafficLight)

if __name__ == '__main__':
    # Test if the model works
    board = Board(10, 10, 10)
    for i in range(10):
        board.step()
        all = board.datacollector.get_model_vars_dataframe()
        print(all.iloc[i]['Grid'])
