import numpy as np

WORLD_HEIGHT = 4
WORLD_WIDTH = 12
GAMMA = 1
EPSILON = 0.1
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
START = [3, 0]
GOAL = [3, 11]

class ENV:
   
    def __init__(self):
        self.START = [3, 0]
        self.GOAL = [3, 11]
        self.WORLD_WIDTH = WORLD_WIDTH
        self.WORLD_HEIGHT = WORLD_HEIGHT
    
    def step(self, s, a):
        i, j = s
        if a == UP:
            next_S = [max(i - 1, 0), j]
        elif a == LEFT:
            next_S = [i, max(j - 1, 0)]
        elif a == RIGHT:
            next_S = [i, min(j + 1, self.WORLD_WIDTH - 1)]
        elif a == DOWN:
            next_S = [min(i + 1, self.WORLD_HEIGHT - 1), j]
        else:
            raise ValueError
        reward = -1
        if next_S[0] == 3 and 1 <= next_S[1] <= 10:
            reward = -100
            next_S = self.START
        terminal = False
        if next_S == self.GOAL:
            terminal = True
        return next_S, reward, terminal


 


    