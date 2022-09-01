from mesa import Agent

class Robot(Agent):
    DIRECTIONS = ['right', 'down', 'left', 'up']
    def __init__(self, unique_id, model, direction=None, position = None, status = "Robot", cargo = False):
        super().__init__(unique_id, model)
        self._direction = self.random.choice(self.DIRECTIONS) if not direction else direction
        self._position = position if not position else self.random.choice(self.model.grid.coord_iter())
        self._status = status
        self._cargo = cargo
        self.next_position = None
        # Initialize in random empty cell
        self.pos = model.grid.find_empty()
        self.model.grid.place_agent(self, (self.pos))
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
        
        # Search an stack if it is already moving a box, otherwise search for a box
        for agent in self.model.schedule.agents:

            if self._cargo:
                if isinstance(agent, Stack):
                    if agent.full:
                        continue
                    # Try to get closer, if not possible, try to move in the same direction
                    
                    #First move on x axis
                    if self.pos[0] < agent.pos[0]:
                        self.direction = "right"
                        break
                    if self.pos[0] > agent.pos[0]:
                        self.direction = "left"
                        break
                    
                    #Then move on y axis
                    if self.pos[1] < agent.pos[1]:
                        self.direction = "down"
                        break
                    if self.pos[1] > agent.pos[1]:
                        self.direction = "up"
                        break

                    # If reached the stack, drop the box
                    self._cargo = False
                    agent.capacity += 1
                    self.binded_box.binded = False
                    break
            else:
                # If not moving a box, search for one
                if isinstance(agent, Box):
                    if agent.binded:
                        continue
                    # Try to get closer, if not possible, try to move in the same direction
                    
                    #First move on x axis
                    if self.pos[0] < agent.pos[0]:
                        self.direction = "right"
                        break
                    if self.pos[0] > agent.pos[0]:
                        self.direction = "left"
                        break
                    
                    #Then move on y axis
                    if self.pos[1] < agent.pos[1]:
                        self.direction = "down"
                        break
                    if self.pos[1] > agent.pos[1]:
                        self.direction = "up"
                        break

                    # If reached the box, bind it
                    agent.binded = True
                    self._cargo = True
                    self.binded_box = agent
                    break

        self.next_position = self.pos[0] + self.dx, self.pos[1] + self.dy
            
    def step(self):
        self.move()

    def advance(self):
        if self.binded_box:
            self.binded_box.pos = self.next_position
        self.model.grid.move_agent(self, self.next_position)

class Stack(Agent):
    def __init__(self, unique_id, model, position = None):
        super().__init__(unique_id, model)
        self._position = position if not position else self.random.choice(self.model.grid.coord_iter())
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
        self.model.grid.place_agent(self, (self.pos))
    def step(self):
        pass

