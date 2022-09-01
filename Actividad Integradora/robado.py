import agentpy as ap
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns
import IPython

class agentePila(ap.Agent):
    def setup(self):
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.cajas = 0
        self.lleno = False
        
    def setup_parameters(self): #Se incluye unicamente para mayor entendimiento
        pass
    
    def aumentarPila(self): #Aumentar numero de cajas en la pila
        self.cajas += 1
    
    def checkPila(Self): #Verificar tamaño de la pila
        if(self.cajas == 5):
            lleno = True
        else:
            lleno = False
    
class agenteCaja(ap.Agent):
    def setup(self):
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.picked = False
       
    def setup_parameters(self): #Se incluye unicamente para mayor entendimiento
        pass
    
    def pickup(self): #Detectar si la caja ya ha sido recolectada
        self.picked = True
        
    def removeCaja(self): #Borrar caja del escenario puesto que esta en la pila
        self.grid.remove_agents(self)

class agenteRobot(ap.Agent):
    def setup(self):
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.move = {"l": (-1,0) ,"r": (1,0),"u": (0,-1),"d": (0,1)}
        self.stuck = False
        self.izquierda = False
        self.encontrePila = False
    
    def setup_parameters(self, velocidad, cajaCarga, posX, posZ,n,m): #Inicializar valores del robot
        self.velocidad = velocidad #bool, solo on y off
        self.cajaCarga = None #bool, carga o no la caja
        self.direccion = True #Verifica que la direccion es viable
        self.n = n #si x < n
        self.m = m #si z < m
        self.posX = self.grid.positions[self][0] #int, se movera en cuadrantes
        self.posZ = self.grid.positions[self][1] #int, se movera en cuadrantes
        self.posY = 0 #0, no se mueve en eje yy
        
    def actualizarVelocidad(self, velocidad): #Cambiarla (prendido -> apagado)
        check = velocidad
        if(check):
            velocidad = False
        else:
            velocidad = True
    
    def actualizarPosicion(self):
        if self.stuck: return #Un robot no se puede mover
        
        if self.cajaCarga: #Cargar caja
            print(self, "CARGO UNA CAJAAAAAAAAA :))))")
            if self.encontrePila:
                if 'r' in self.move:
                    print(self, "Estoy caminando derecha")
                    self._move(1,0)
                else:
                    self._randomMove()
                #Ve hacia la derecha
            elif self.izquierda:
                if 'u' in self.move:
                    print(self, "Estoy caminando arriba")
                    self._move(0,-1)
                else:
                    self._move(1,0)
                    self._move(0,-1)
                    self._move(0,-1)
                    self._move(-1,0)
                    self._randomMove()
            else:
                if 'l' in self.move:
                    print(self, "Estoy caminando izquierda")
                    self._move(-1,0)
                else:
                    self._randomMove()
        else:
            self._randomMove() #Movimientos random hacia las 4 direcciones(arriba, abajo, izquieda, derecha)
        #izquiera
        #self.grid.move_by(self, (-1,0))
        #arriba
        #self.grid.move_by(self, (0,1))
        #abajo
        #self.grid.move_by(self, (0,-1))
            
    def _move(self,horizontal,vertical): #Mover en X o Z
        self.grid.move_by(self, (horizontal,vertical))
        self.posX += horizontal
        self.posZ += vertical
            
    def _randomMove(self): #Mover random
        if self.cajaCarga:
            print(self, "Me bloquearon voy random :((")
        
        keys = []
        for k in self.move:
            keys.append(k)
        #print(self,"possible moves: ", keys)
        
        number = random.randint(0,len(keys)-1)
        direction = keys[number]
        #print(self,"choice:", direction, number)
        #self.move = {"l": (-1,0) ,"r": (1,0),"u": (0,-1),"d": (0,1)}

        #derecha
        self._move(self.move[direction][0] , self.move[direction][1])

    def depositaCaja(self, pila): #Quitar caja en el escenario, aumentarla en la pila
        print('PUSE LA CAJAAAAAAAAAAAAA', self.cajaCarga)
        pila.aumentarPila()
        self.cajaCarga.removeCaja()
        self.cajaCarga = None
        
    def detectarObstaculos(self): #Detectar diferentes obstaculos del escenario
        self.izquierda = False
        self.encontrePila = False
        self.move = {"l": (-1,0) ,"r": (1,0),"u": (0,-1),"d": (0,1)} #Crear un arreglo de 4 direcciones
        if self.posX == self.n-1:
            self.move.pop('r') #Quitar la derecha del arreglo
        if self.posX == 0:
            self.move.pop('l') #Quitar la izquierda del arrego
            self.izquierda = True
        if self.posZ == self.m-1: #Quitar abajo del arreglo
            self.move.pop('d')
        if self.posZ == 0:
            self.move.pop('u') #Quitar arriba del arreglo
        neighbors = self.grid.neighbors(self)
        for n in neighbors:
            if self.cajaCarga and n.type == "agentePila": #Checar si choco con una pila
                if n.cajas < 5: #Si la caja cabe en la pila ponerlo
                    self.depositaCaja(n)
                else:
                    self.encontrePila = True #Encontro zona de pilas, busca donde dejarla
            deletePos = []
            for pos in self.move:
                if self.grid.positions[n] == (self.posX + self.move[pos][0], self.posZ + self.move[pos][1]): #Obstaculo esta en donde se quiere mover
                    deletePos.append(pos)
            for p in deletePos:
                self.move.pop(p)
        if len(self.move) == 0:
            stuck = True #Un robot quedo atorado
            print('No me puedo mover :(((')
        else:
            stuck = False
            
        #Pared
        #Cajas
        #CantCajasPila
        #Robot
        
    def detectarCargaCaja(self):
        neighbors = self.grid.neighbors(self)
        
        for n in neighbors:
            if(n.type == "agenteCaja" and not n.picked): #Agarrar la caja encontrada
                print("PICK LA CAJAAAAAAAAAAAAAAA", n)
                self.cajaCarga = n
                n.pickup()
                break
    
class modeloRobot(ap.Model):
    def setup(self):
        s = self.p.size
        n = 5 #Num robots
        self.n_caja = random.randrange(self.p.size - 1) + 1 #Num random de cajas
        pilas = self.n_caja // 5 #Cantidad de pilas a generar segun Num cajas
        if(self.n_caja % 5 != 0):
            pilas +=1 #Crear pilas
        self.grid = ap.Grid(self, (s, s), track_empty=True)
        self.agents = ap.AgentList(self, n, agenteRobot)
        self.cajas = ap.AgentList(self,self.n_caja, agenteCaja)
        self.pilas = ap.AgentList(self,pilas, agentePila)
        self.grid.add_agents(self.pilas,random = False, empty = True, positions = [(i,0) for i in range(pilas)]) #Colocar pilas en la fila superior
        self.grid.add_agents(self.agents, random = True, empty = True) #Colocar robots random
        self.grid.add_agents(self.cajas, random = True, empty = True) #Colocar cajas random
        
        self.agents.setup_parameters(self.p.velocidad, self.p.cajaCarga, self.p.posX, self.p.posZ,s,s) #Inicializar robots con variables definidas
    
    def step(self):
        for i in self.grid.positions:
            print (i, self.grid.positions[i])
            
        self.agents.detectarCargaCaja()
        self.agents.detectarObstaculos()
        self.agents.actualizarPosicion()
        
    def update(self):
        #self.agents.actualizarVelocidad(self.p.velocidad)
        cajitas = 0
        for pila in self.pilas:
            cajitas += pila.cajas
        if cajitas == self.n_caja:
            self.stop()
    
    def end(self):
        self.report('Cantidad de cajas iniciales: ', self.n_caja)
        self.report('Cantidad de pilas: ', len(self.pilas))

parameters = {
    'size': 10,
    'velocidad': 0,
    'cajaCarga': 0,
    'posX': 0,
    'posZ': 0,
    'steps': 1000,
    'n_groups': 3
}

model = modeloRobot(parameters)

results = model.run()

print("INFORMACIÓN")
print(results)
print("REPORTE EN RESULT: ")
print(results.reporters)
print("REPORTE DE LA SIMULACIÓN: ")
print(results.info)