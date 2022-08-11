# Commented out IPython magic to ensure Python compatibility.
# 'Model' sirve para definir los atributos a nivel del modelo, maneja los agentes
# 'Agent' es la unidad atómica y puede ser contenido en múltiples instancias en los modelos
from mesa import Agent, Model 

# 'SingleGrid' sirve para forzar a un solo objeto por celda (nuestro objetivo en este "juego")
from mesa.space import SingleGrid

# 'SimultaneousActivation' habilita la opción de activar todos los agentes de manera simultanea.
from mesa.time import SimultaneousActivation

# 'DataCollector' permite obtener el grid completo a cada paso (o generación), útil para visualizar
from mesa.datacollection import DataCollector

# 'matplotlib' lo usamos para graficar/visualizar como evoluciona el autómata celular.
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2 ** 128

# Definimos los siguientes paquetes para manejar valores númericos: 'numpy' & 'pandas'
import numpy as np
import pandas as pd

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime

# Tamaño del espacio
m = 5
n = 5

# Número de



class cleanAgent(Agent):
    def __init__(self,unique_id,x,bateria,model):
        super().__init__(unique_id, model)
        self.next_state = None
        self.bateria = bateria
    def advance(self):
        """
        Define el nuevo estado calculado del método step.
        """
        self.live = self.next_state
    def step(self,board):
        # Get current cell
        current_cell = self.model.grid.get_cell_list_contents([self.pos])[0]
        # If current cell is dirty, clean it
        if current_cell.next_state:
            self.clean(current_cell)
            self.next_state = self.next_state
            return True
        # If current cell is clean, move to a random neighbour
        else:
            self.move(board)
            self.next_state = False
            return True
    def clean(self,cell):
        cell.next_state = False
        self.next_state = False
        return True
    def move(self,board):
        neighbours = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False)
        random.shuffle(neighbours)
        self.pos = neighbours[0]
class cell(Agent):
    def __init__(self,unique_id,model,state):
        super().__init__(unique_id, model)
        self.next_state = None
        self.pos = unique_id
        self.state = state
class Board(Model):
