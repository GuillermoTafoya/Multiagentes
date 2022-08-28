from mesa import Agent
import DQN_only_agent from IA_Trafficlight

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

        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=False, radius=max(self.model.width, self.model.height))
        
        for neighbour in neighbours:

            if isinstance(neighbour, TrafficLight):
                if neighbour.state == True and self.opositeDirections(neighbour.direction, self.direction):
                    # stop
                    if (self.direction == 'down') and neighbour.pos[1] - self.pos[1] == 1:
                        return
                    if (self.direction == 'up') and self.pos[1] - neighbour.pos[1] == 1:
                        return
                    if (self.direction == 'right') and neighbour.pos[0] - self.pos[0] == 1:
                        return
                    if (self.direction == 'left') and self.pos[0] - neighbour.pos[0] == 1:
                        return
            # Try stopping if there is another car in the way
            if isinstance(neighbour, Car):
                # Check collision with cars
                if self.pos == neighbour.pos:
                    self.alive = False
                    neighbour.alive = False
                    return
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
    

        # Move
        next_pos = (self.pos[0] + self.dx, self.pos[1] + self.dy)
        if self.model.grid.out_of_bounds(next_pos):
            self.alive = False
            return
        self.next_pos = next_pos
        self.model.grid.move_agent(self, next_pos)

class TrafficLight(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, state = True, direction = None):
        super().__init__(unique_id, model)
        self.state = state
        self.direction = direction

    def step(self):
        self.state = self.makeDecision()

    def makeDecision(self):
        return DQN_only_agent.makeDecision(self)


        

class Road(Agent):
    """
    Obstacle agent. Just to add obstacles to the grid.
    """
    def __init__(self, unique_id, model, colour = None):
        super().__init__(unique_id, model)
        self.colour = colour if colour else "olive"
    def step(self):
        pass