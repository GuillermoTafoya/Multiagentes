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
from model import Cell 

class Auto(Agent):
    def __init__(self,unique_id,model,axis):
        super().__init__(unique_id, model)
        self.next_state = None
        self.moves = 0
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
    def make_decision(self,neighbours):
        """
        Define la decisión del agente.
        """
        return random.choice(neighbours) # Algoritmo de decisión aleatorio.
