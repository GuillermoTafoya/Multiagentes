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
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        for obj in cell_content:
            if isinstance(obj, CleanAgent):
                grid[x][y] = 2
            elif isinstance(obj, Cell):
                grid[x][y] = obj.state
    return grid
class CleanAgent(Agent):
    def __init__(self,unique_id,model):
        super().__init__(unique_id, model)
        self.next_state = None
        self.moves = 0
        #self.bateria = bateria
    def advance(self):
        """
        Define el nuevo estado calculado del método step.
        """
        neighbours = self.model.grid.get_neighbors(
        self.pos, 
        moore = True,
        include_center = True)
        for neighbour in neighbours:
            if isinstance(neighbour, Cell) and self.pos == neighbour.pos:
                neighbour.state = neighbour.next_state
                break
        self.model.grid.move_agent(self, self.next_state)
    def step(self):
        neighbours = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False)
        for neighbour in neighbours:
            if isinstance(neighbour, Cell)  and self.pos == neighbour.pos:
                if neighbour.state:
                    neighbour.next_state = 0
                    self.next_state = self.pos
            else:
                self.moves += 1
                neighbours_2 = self.model.grid.get_neighborhood(
                self.pos, 
                moore = True,
                include_center = False)
                neighbour.next_state = 0
                ### Implement shortest path ###
                self.next_state = self.make_decision(neighbours_2)
            break
    def make_decision(self,neighbours):
        """
        Define la decisión del agente.
        """
        return random.choice(neighbours) # Algoritmo de decisión aleatorio.
class Cell(Agent):
    def __init__(self,unique_id,model,state):
        super().__init__(unique_id, model)
        self.next_state = None
        self.pos = unique_id
        # 1 for transitable, 0 for non-transitable
        self.state = state
class Board(Model):
    def __init__(self,m,n,num_agentes,p_sucias):
        self.num_agentes = num_agentes
        self.schedule = SimultaneousActivation(self)
        self.grid = MultiGrid(m,n,False) # Bool defines toroidal boundary
        num_celdas_sucias = int(m*n*p_sucias)
        for (content,x,y) in self.grid.coord_iter():
            num = random.randint(0,1)
            if num == 1 and num_celdas_sucias > 0:
                a = Cell((x,y), self, 1)
                num_celdas_sucias -= 1
            else:
                a = Cell((x,y), self, 0)
            self.grid.place_agent(a,(x,y))
            self.schedule.add(a)
        for id in range(num_agentes):
            r = CleanAgent(id, self)
            self.grid.place_agent(r, (1,1))
            self.schedule.add(r)

        self.datacollector = DataCollector(
            model_reporters = {"Grid":get_grid})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def allClean(self):
        for (content, x, y) in self.grid.coord_iter():
            for obj in content:
                if isinstance(obj, Cell) and obj.state == 1:
                    return False
        return True

    def numDirty(self):
        cantidad_celdas_sucias = 0
        for (content, x, y) in self.grid.coord_iter():
            for obj in content:
                if isinstance(obj, Cell) and obj.state == 1:
                    cantidad_celdas_sucias = cantidad_celdas_sucias + 1
        return cantidad_celdas_sucias

    def count_moves(self, M, N, num_agentes):
        cont = 0
        for i in range (N*M, N*M + num_agentes):
            cont = cont + self.schedule.agents[i].moves
        return cont