from mesa import Agent

class Car(Agent):
    DIRECTIONS = ['right', 'down', 'left', 'up']

    def __init__(self, unique_id, model, colour=None, direction=None):
        super().__init__(unique_id, model)
        self.colour = colour
        self.dx = 0
        self.dy = 0
        self._direction = self.random.choice(self.DIRECTIONS) if not direction else direction
        self.direction = self._direction
        self.alive = True
        self.successful_trip = False
        self.stopped = False
        self.next_pos = unique_id

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = direction
        if self._direction == 'up':
            self.dx, self.dy = 0, -1
            return
        if self._direction == 'down':
            self.dx, self.dy = 0, 1
            return
        if self._direction == 'right':
            self.dx, self.dy = 1,0
            return
        if self._direction == 'left':
            self.dx, self.dy = -1,0
            return
        raise(ValueError('Invalid direction'))

    def opositeDirections(self, direction1, direction2):
        if direction1 == 'up' and direction2 == 'down':
            return True
        if direction1 == 'down' and direction2 == 'up':
            return True
        if direction1 == 'right' and direction2 == 'left':
            return True
        if direction1 == 'left' and direction2 == 'right':
            return True
        return False

    def step(self):
        """
        Defines how the model interacts within its environment.
        """
        
        # Check if the agent is alive
        if not self.alive:
            return
        
        # Move
        next_pos = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        
        
        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=max(self.model.width, self.model.height))
        
        for neighbour in neighbours:

            if isinstance(neighbour, TrafficLight):
                if neighbour.state == True and self.opositeDirections(neighbour.direction, self.direction):
                    # stop
                    if (self.direction == 'down') and neighbour.pos[1] - self.pos[1] == 1:
                        self.stopped = True
                        return
                    if (self.direction == 'up') and self.pos[1] - neighbour.pos[1] == 1:
                        self.stopped = True
                        return
                    if (self.direction == 'right') and neighbour.pos[0] - self.pos[0] == 1:
                        self.stopped = True
                        return
                    if (self.direction == 'left') and self.pos[0] - neighbour.pos[0] == 1:
                        self.stopped = True
                        return
            # Try stopping if there is another car in the way
            if isinstance(neighbour, Car):
                if (self.direction == neighbour.direction == 'down') and neighbour.pos[1] - self.pos[1] == 1:
                    if neighbour.pos[0] == self.pos[0]:
                        return
                elif (self.direction == neighbour.direction == 'up') and self.pos[1] - neighbour.pos[1] == 1:
                    if neighbour.pos[0] == self.pos[0]:
                        return
                elif (self.direction == neighbour.direction == 'right') and neighbour.pos[0] - self.pos[0] == 1:
                    if neighbour.pos[1] == self.pos[1]:
                        return
                elif (self.direction == neighbour.direction == 'left') and self.pos[0] - neighbour.pos[0] == 1:
                    if neighbour.pos[1] == self.pos[1]:
                        return
                
                """
                # Check collision with cars
                if self.next_pos == neighbour.next_pos and neighbour is not self and neighbour.stopped == self.stopped == False:
                    self.alive = False
                    neighbour.alive = False
                    return
                """
        
        self.stopped = False
        if self.model.grid.out_of_bounds(next_pos):
            self.successful_trip = True
            return
        self.next_pos = next_pos
        self.model.grid.move_agent(self, next_pos)

class TrafficLight(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, state = True, timeToChange = 10, direction = None, delay = 0):
        super().__init__(unique_id, model)
        self.state = state
        self.timeToChange = timeToChange
        self._timeToChange = timeToChange
        self.direction = direction
        self._delay = delay
        self.delay = 0

    def step(self):
        if self.state and self.delay < self._delay:
            self.delay += 1
            return
        if self.state and self.delay >= self._delay:
            self._delay = 0
            self.timeToChange -= 1
            if self.timeToChange == 0:
                self.state = False
                self.timeToChange = self._timeToChange
        else:
            self.timeToChange -= 1
            if self.timeToChange == 0:
                self.state = True
                self.timeToChange = self._timeToChange
        

class Road(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, colour = None):
        super().__init__(unique_id, model)
        self.colour = colour if colour else "olive"
    def step(self):
        pass