import math
import random

EPSILON_CHANCE = 10**-3
SAME_STEP_SIZE_IN_A_ROW = 80

MAX_ALPHA = 10**4
class Absolute_Step():
    def __init__(self, options = None):
        self.counter = 0
        self.last_direction = 0
        self.options = options if options is not None else [5 ** i for i in range(-3, 2)]

    def do_step(self, alpha, direction):
        if direction == self.last_direction:
            self.counter += 1
        else:
            self.counter = 0
        self.last_direction = direction

        idx = math.floor(self.counter / SAME_STEP_SIZE_IN_A_ROW)

        # with small probability do a big step
        if random.uniform(0,1) <= EPSILON_CHANCE:
            # if idx > len(self.options):
            #     # this means we are at the end & got the epsilon chance, let's reset the search
            #     return random.normalvariate(0,1)
            alpha = alpha + (direction * 1)
        else:
            idx = idx if idx < len(self.options) else -1
            alpha = alpha + (direction * self.options[idx])
        return alpha if alpha < MAX_ALPHA else alpha / 2
        # return alpha + (direction * step_size)

sign = lambda x: 1 if x >= 0 else -1

class Relative_Step():
    def __init__(self, options = None):
        self.counter = 0
        self.last_direction = 0
        self.options = options if options is not None else [0.01,0.05, 0.1]#, 0.2,0.3]
        # self.options = [0.4, 0.9]

    def do_step(self, alpha, direction):
        if direction == self.last_direction:
            self.counter += 1
            if self.counter >= len(self.options):
                self.counter = len(self.options) - 1
        else:
            self.counter = 0
        self.last_direction = direction

        if abs(alpha) > 0.001:
            alpha =  alpha + (direction * alpha * self.options[self.counter] * sign(alpha))  # do step size 0.3 to the next direction
        else:
            alpha = 0.5 * direction

        return alpha if alpha < MAX_ALPHA else alpha / 2

        # return alpha + direction * (
        #     self.options[self.counter] if self.counter < len(self.options) else self.options[-1])
        # return alpha + (direction * step_size)