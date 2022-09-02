from random import random
from mesa import Agent

class Robot(Agent):
    
    def __init__(self, unique_id, model, direction=None):
        super().__init__(unique_id, model)
        self.DIRECTIONS = ['right', 'down', 'left', 'up']
        self._direction = self.random.choice(self.DIRECTIONS) if not direction else direction
        self.dx = 0
        self.dy = 0
        self.cargo = False
        self.next_position = None
        self.binded_box = None
        self.objective = None
        
        # Initialize in random empty cell
        self.pos = model.grid.find_empty()
    @property
    def direction(self):
        return self._direction
    @property
    def status(self):
        return self._status
    @direction.setter
    def direction(self, direction):
        self._direction = direction
        if self._direction == 'up':
            self.dx, self.dy = -1, 0
            return
        if self._direction == 'down':
            self.dx, self.dy = 1, 0
            return
        if self._direction == 'right':
            self.dx, self.dy = 0, 1
            return
        if self._direction == 'left':
            self.dx, self.dy = 0, -1
            return
        raise(ValueError('Invalid direction'))

    def move(self):

        if self.objective:
            if self.cargo:
                # If the stack became full while the robot was moving
                if self.objective.full:
                    self.objective = None
                    return
            #First move on x axis
            if self.pos[1] < self.objective.pos[1]:
                self.direction = "right"
                return
            if self.pos[1] > self.objective.pos[1]:
                self.direction = "left"
                return
            
            #Then move on y axis
            if self.pos[0] < self.objective.pos[0]:
                self.direction = "down"
                return
            if self.pos[0] > self.objective.pos[0]:
                self.direction = "up"
                return

            if self.cargo:
                # If reached the stack, drop the box
                self.cargo = False
                self.objective.capacity += 1
                self.binded_box.delievered = True
                self.binded_box = None
                self.objective = None
                self.direction = "left"
            else:
                # If reached the box, bind it
                self.objective.binded = True
                self.cargo = True
                self.binded_box = self.objective
                self.objective = None
                return

        # Search an stack if it is already moving a box, otherwise search for a box
        for agent in self.model.schedule.agents:
            if self.cargo:
                if isinstance(agent, Stack):
                    if agent.full:
                        continue
                    self.objective = agent
                    return
            else:
                #self.direction = self.random.choice(self.DIRECTIONS)
                # If not moving a box, search for one
                if isinstance(agent, Box) and not agent.targeted and not agent.binded and not agent.delievered:
                    self.objective = agent
                    agent.targeted = True
                    return

        
            
    def step(self):
        self.move()
        next_position = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        # If the next position has a robot or a wall, change direction
        """if not self.model.grid.is_cell_empty(next_position):
            # Check if the next position is a robot
            if isinstance(self.model.grid.get_cell_list_contents(next_position)[0], Robot):
                # If the next position is a robot, change direction
                self.direction = self.random.choice(self.DIRECTIONS)
                return"""
        # If the next position is out of the grid, change direction
        while (self.model.grid.out_of_bounds(next_position)):
            self.direction = self.random.choice(self.DIRECTIONS)
            next_position = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        self.next_position = next_position



    def advance(self):
        if self.binded_box:
            self.model.grid.move_agent(self.binded_box, self.next_position)
        self.model.grid.move_agent(self, self.next_position)

class Stack(Agent):
    def __init__(self, unique_id, model, position = None):
        super().__init__(unique_id, model)
        self._position = position if position else self.random.choice(self.model.grid.coord_iter())
        self.full = False
        self._capacity = 0
    @property
    def capacity(self):
        return self._capacity
    @capacity.setter
    def capacity(self, capacity):
        self._capacity = capacity
        if self._capacity == 5:
            self.full = True
        else:
            self.full = False
    def step(self):
        pass

class Box(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pos = self.model.grid.find_empty()
        self.binded = False
        self.delievered = False
        self.targeted = False
        self.model.grid.place_agent(self, (self.pos))
    def step(self):
        pass

