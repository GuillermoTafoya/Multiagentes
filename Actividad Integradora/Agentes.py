from mesa import Agent

class Robot(Agent):
    DIRECTIONS = ['right', 'down', 'left', 'up']
    def __init__(self, unique_id, model, direction=None, position = None, status = "Robot", cargo = False):
        super().__init__(unique_id, model)
        self._direction = self.random.choice(self.DIRECTIONS) if not direction else direction
        self._position = position if not position else self.random.choice(self.model.grid.coord_iter())
        self._status = status
        self._cargo = cargo
        # Initialize in random empty cell
        self.position_x, self.position_y = model.grid.find_empty()
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
        posible_steps = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        new_position = self.random.choice(posible_steps)
        self.model.grid.move_agent(self, new_position)
    def pickup_box(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            for inhabitants in cellmates:
                if self._cargo == 0 and isinstance(inhabitants, Box):
                    self._cargo = 1
                    self._status = "Robot with cargo"
                    self.model.grid.remove_agent(inhabitants)
    def organize_box(self):
        if self._cargo:
            pass
            
    def step(self):
        self.move()
        self.pickup_box()
        

class Stack(Agent):
    def __init__(self, unique_id, model, position = None):
        super().__init__(unique_id, model)
        self._position = position if not position else self.random.choice(self.model.grid.coord_iter())
    def step(self):
        pass

class Box(Agent):
    max_position_x = length
    max_position_y = height
    def __init__(self, unique_id, model, position_x = None, position_y=None, position_z = None ):
        super().__init__(unique_id, model)
        self._position_x = self.randint(0, max_position_x) if not position_x else position_x
        self._position_y = self.randint(0, max_position_y) if not position_y else position_y
        self._position_z = 0 if not position_z else position_z
    @property
    def position(self):
        return self._position

