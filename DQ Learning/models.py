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
    json = "{"
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell        
        for agent in cell_content:
            # Save relevant agents to json
            if isinstance(agent, Car):
                json += f'"{agent.unique_id}":{{"x":{x},"y":{y},"speed":{agent.dx,agent.dy},"direction":"{agent.direction}"}},'
                if agent.colour == 'white':
                    grid[x][y] = 6
                elif agent.colour == 'blue':
                    grid[x][y] = 7
            
            elif isinstance(agent, TrafficLight):
                if agent.state == True: # Red
                    grid[x][y] = 2
                else: # Green
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
    json = json[:-1] + "}"
    # Append json to file
    with open("data.txt", "a") as file:
        file.write(json + "\n")
    return grid


class Board(Model):
    def __init__(self, width, height, seed=None, spawn_rate = 1, max_spawn_batch = 1):
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid})
        random.seed(seed if seed is not None else time.time())
        self.carID = 2
        self.spawn_rate = spawn_rate
        self.crashes = 0
        self.successful_trips = 0
        self.max_spawn_batch = max_spawn_batch
        self.create_agents()

    
    def step(self):
        
        if self.schedule.steps % self.spawn_rate == 0:
            for _ in range(random.randint(1, self.max_spawn_batch)):
                self.spawn_random_car()
        
        self.schedule.step()
        self.datacollector.collect(self)
        
        # Check if there are cars that reached the destination
        for agent in self.schedule.agents:
            if isinstance(agent, Car):

                if agent.successful_trip:
                    self.successful_trips += 1
                    self.schedule.remove(agent)
                    self.grid.remove_agent(agent)
                    del agent
                    continue

                neighbours = self.grid.get_neighbors(agent.pos, moore=False, include_center=False)
                for neighbour in neighbours:
                    if isinstance(neighbour, Car):
                        # Check collision with cars
                        if agent.next_pos == neighbour.next_pos and neighbour is not agent and neighbour.stopped == agent.stopped == False:
                            self.crashes += 1
                            self.schedule.remove(agent)
                            self.grid.remove_agent(agent)
                            
                            self.schedule.remove(neighbour)
                            self.grid.remove_agent(neighbour)
                            
                            del agent
                            del neighbour
                            break

    def spawn_random_car(self):
        direction = random.choice(["down", "right"])
        car = Car(self.carID, self, direction = direction, colour = 'white' if direction == 'down' else 'blue')
        self.carID += 1
        x = random.randint(self.width // 3, self.width * 2 // 3) if direction == "down" else 0
        y = 0 if direction == "down" else random.randint(self.height // 3, self.height * 2 // 3)
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

        # Create traffic lights
        # Only two traffic lights, one from up to down and one from right to left
        trafficLight = TrafficLight(0, self, state=False, timeToChange=self.width, direction = "left", delay = self.height)
        self.grid.place_agent(trafficLight, (self.width // 3, self.height // 3 * 2))
        self.schedule.add(trafficLight)

        trafficLight = TrafficLight(1, self, state=True, timeToChange=self.height, direction = "up", delay = self.width)
        self.grid.place_agent(trafficLight, (self.width // 3 * 2, self.height // 3))
        self.schedule.add(trafficLight)

if __name__ == '__main__':
    # Test if the model works
    board = Board(10, 10, 10)
    for i in range(10):
        board.step()
        all = board.datacollector.get_model_vars_dataframe()
        print(all.iloc[i]['Grid'])
