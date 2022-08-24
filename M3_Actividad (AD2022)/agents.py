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
        
        self.next_pos = unique_id

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = direction
        if self._direction == 'up':
            self.dx, self.dy = 0, 1
            return
        if self._direction == 'down':
            self.dx, self.dy = 0, -1
            return
        if self._direction == 'right':
            self.dx, self.dy = 1, 0
            return
        if self._direction == 'left':
            self.dx, self.dy = -1, 0
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

        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=False, radius=max(self.model.width, self.model.height))
        
        for neighbour in neighbours:

            if isinstance(neighbour, TrafficLight):
                if neighbour.state == True and self.opositeDirections(neighbour.direction, self.direction):
                    # stop
                    if (self.direction == 'up' or self.direction == 'down') and abs(neighbour.pos[1] - self.pos[1]) == 1:
                        return
                    if (self.direction == 'right' or self.direction == 'left') and abs(neighbour.pos[0] - self.pos[0]) == 1:
                        return
            # Try stopping if there is another car in the way
            if isinstance(neighbour, Car):
                if (self.direction == 'up' or self.direction == 'down') and abs(neighbour.pos[1] - self.pos[1]) == 1:
                    if neighbour.pos[1] == self.pos[1] and neighbour.direction == self.direction:
                        return
                elif (self.direction == 'right' or self.direction == 'left') and abs(neighbour.pos[0] - self.pos[0]) == 1:
                    if neighbour.pos[0] == self.pos[0] and neighbour.direction == self.direction:
                        return

            # Check collision with cars
            if isinstance(neighbour, Car) and not (self is neighbour):
                if self.next_pos == neighbour.next_pos:
                    self.alive = False
                    neighbour.alive = False
                    return

        # Move
        next_pos = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        self.next_pos = next_pos
        self.model.grid.move_agent(self, next_pos)

class TrafficLight(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, state = False, timeToChange = 10, direction = None):
        super().__init__(unique_id, model)
        self.state = state
        self.timeToChange = timeToChange
        self.direction = direction

    def step(self):
        if self.model.schedule.steps % self.timeToChange == 0:
            self.state = not self.state

class Road(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, colour = None):
        super().__init__(unique_id, model)
        self.colour = colour if colour else "olive"
    def step(self):
        pass