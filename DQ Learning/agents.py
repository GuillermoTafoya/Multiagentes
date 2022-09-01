#Aiuda

from re import S
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
        if not self.alive:
            return
        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=max(self.model.width, self.model.height))
        for neighbour in neighbours:

            if isinstance(neighbour, TrafficLight):
                if neighbour.state == True and self.opositeDirections(neighbour.direction, self.direction):
                    # stop
                    if (self.direction == 'down') and neighbour.pos[1] - self.pos[1] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                    if (self.direction == 'up') and self.pos[1] - neighbour.pos[1] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                    if (self.direction == 'right') and neighbour.pos[0] - self.pos[0] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                    if (self.direction == 'left') and self.pos[0] - neighbour.pos[0] == 1:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
        self.stopped = False
        self.next_pos = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        
        

        
        
    def advance(self):
        if self.stopped:
            self.next_pos = self.pos
            return
        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=max(self.model.width, self.model.height))
        
        
        for neighbour in neighbours:
            # Try stopping if there is another car in the way
            if isinstance(neighbour, Car):
                if (self.direction == neighbour.direction == 'down') and neighbour.pos[1] - self.pos[1] == 1 and neighbour.stopped:
                    if neighbour.pos[0] == self.pos[0]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                elif (self.direction == neighbour.direction == 'up') and self.pos[1] - neighbour.pos[1] == 1 and neighbour.stopped:
                    if neighbour.pos[0] == self.pos[0]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                elif (self.direction == neighbour.direction == 'right') and neighbour.pos[0] - self.pos[0] == 1 and neighbour.stopped:
                    if neighbour.pos[1] == self.pos[1]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                elif (self.direction == neighbour.direction == 'left') and self.pos[0] - neighbour.pos[0] == 1 and neighbour.stopped:
                    if neighbour.pos[1] == self.pos[1]:
                        self.stopped = True
                        self.next_pos = self.pos
                        return
                
                
                # Check collision with cars
                if self.next_pos == neighbour.next_pos and neighbour is not self and self.direction != neighbour.direction and neighbour.stopped == self.stopped == False:
                    
                    self.alive = False
                    neighbour.alive = False
                    #self.next_pos = self.pos
                    return

        
        
        # Check if the car has reached the goal
        if self.direction == 'down' and self.next_pos[1] == self.model.height - 1:
            self.successful_trip = True
        elif self.direction == 'up' and self.next_pos[1] == 0:
            self.successful_trip = True
        elif self.direction == 'right' and self.next_pos[0] == self.model.width - 1:
            self.successful_trip = True
        elif self.direction == 'left' and self.next_pos[0] == 0:
            self.successful_trip = True
        self.model.grid.move_agent(self, self.next_pos)

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

    #def advance(self) -> None:
        
        

class Road(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, colour = None):
        super().__init__(unique_id, model)
        self.colour = colour if colour else "olive"
    def step(self):
        pass