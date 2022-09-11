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
from Agentes import Robot, Model

class Garage(Model):
    def __init__(self, R, B, width, height, lenght):
        self._num_robots = R
        self._num_boxes = B
        self._grid = MultiGrid(width, height, True)
        self._schedule = SimultaneousActivation(self)

        for i in range(self._num_robots):
            robot = Robot(i, self)
            self.add.agent(robot, self.random.choice(self.grid.coord_iter()))

