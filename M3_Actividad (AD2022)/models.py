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
                """
                # Delete car if it is out of the grid
                if (agent.direction == "left" and x >= model.grid.width-2) or (agent.direction == "right" and x < 2) or (agent.direction == "up" and 2) or (agent.direction == "down" and y >= model.grid.height-2):
                    del agent
                    continue
                # Delete car if it is not alive
                if not agent.alive:
                    del agent
                    continue
                """
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
    def __init__(self, width, height, N, seed=None, spawn_rate = 1):
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid})
        random.seed(seed if seed is not None else time.time())
        self.N = N
        self.carID = 2
        self.spawn_rate = spawn_rate
        self.create_agents()

    
    def step(self):
        
        
        
        if self.schedule.steps % self.spawn_rate == 0:
            self.spawn_random_car()
        self.datacollector.collect(self)
        self.schedule.step()
        # Delete car if it is not alive
        for agent in self.schedule.agents:
            if isinstance(agent, Car) and not agent.alive:
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
                del agent

    def spawn_random_car(self):
        direction = random.choice(["down", "right"])
        car = Car(self.carID, self, direction = direction, colour = 'white' if direction == 'down' else 'blue')
        self.carID += 1
        x = random.randint(self.width // 3, self.width * 2 // 3) if direction == "down" else random.randint(0, 1)
        y = random.randint(0, 1) if direction == "down" else random.randint(self.height // 3, self.height * 2 // 3)
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
            if x < self.width // 3 or x > self.width * 2 // 3:
                if y < self.height // 3 or y > self.height * 2 // 3:
                #print("1. Added road at", x, y)
                    road = Road((x,y), self, colour = "olive")
                    self.grid.place_agent(road, (x, y))
                    self.schedule.add(road)

            """
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
            """
        # Create traffic lights
        # Only two traffic lights, one from up to down and one from right to left
        trafficLight = TrafficLight(0, self, timeToChange=self.width // 3, direction = "left")
        self.grid.place_agent(trafficLight, (self.width // 3, self.height // 3 * 2))
        self.schedule.add(trafficLight)

        trafficLight = TrafficLight(1, self, timeToChange=self.height // 3, direction = "up")
        self.grid.place_agent(trafficLight, (self.width // 3 * 2, self.height // 3))
        self.schedule.add(trafficLight)

if __name__ == '__main__':
    # Test if the model works
    board = Board(10, 10, 10)
    for i in range(10):
        board.step()
        all = board.datacollector.get_model_vars_dataframe()
        print(all.iloc[i]['Grid'])
