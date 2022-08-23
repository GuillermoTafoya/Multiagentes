from mesa import Agent

class Car(Agent):
    DIRECTIONS = ['right', 'down', 'left', 'up']

    def __init__(self, unique_id, model, colour=None, direction=None):
        super().__init__(unique_id, model)
        self.colour = colour
        self._direction = self.random.choice(self.DIRECTIONS) if not direction else direction
        self.alive = True

    @property
    def direction(self):
        return self._direction

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

# ......... more code ............

    def advance(self) -> None:
        """
        Defines a new state calculating the step of the model
        """
        neighbours = self.model.grid.get_neighbors(
        self.pos,
        moore = True,
        include_center = True)
        for neighbour in neighbours:
            if isinstance(neighbour, Car) and self.pos == neighbour.pos:
                # collision
                neighbour.alive = False
                self.alive = False
                break
            if isinstance(neighbour, TrafficLight) and neighbour.state == True and self.opositeDirections(neighbour.direction, self.direction):
                # stop
                return
        self.model.grid.move_agent(self, self.next_state)

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
        next_pos = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        if self.model.grid.out_of_bounds(next_pos):
            self.next_pos = self.model.grid.torus_adj(next_pos)
        else:
            self.next_pos = next_pos

        # ......... more code ............

class TrafficLight(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, state = False, timeToChange = 10):
        super().__init__(unique_id, model)
        self.state = state
        self.timeToChange = timeToChange

    def step(self):
        # if self.model.schedule.steps % self.timeToChange == 0:
        #     self.state = not self.state
        pass
class Road(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, direction= "Left"):
        super().__init__(unique_id, model)
        self.direction = direction

    def step(self):
        pass