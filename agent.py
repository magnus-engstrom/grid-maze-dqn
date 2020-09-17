import random
import numpy as np
from collections import deque

class Agent():
    action_space = [0,1,2,3]
    memory = deque(maxlen=5000)
    max_age = 99
    
    def __init__(self, start_pos, seed=1):
        random.seed(seed)
        np.random.seed(seed)
        self.memory.clear()
        self.x = start_pos[1]
        self.y = start_pos[0]
        self.found_targets = 0
        self.current_age = 0
        self.position = (self.y, self.x)
        self.total_reward = 0
        self.oldx = 0
        self.oldy = 0
        self.action = 0
        self.total_age = 0

    def move(self, suggested_move=None):
        self.oldx = self.x
        self.oldy = self.y
        self.ai = True
        self.current_age += 1
        self.total_age += 1
        self.targets_per_action = self.found_targets / self.total_age
        if suggested_move == None or self.current_age > self.max_age:
            self.action = self._random_action()
        else:
            self.action = suggested_move
        if self.action == 0:
            self.x += 1
        if self.action == 1:
            self.x -= 1
        if self.action == 2:
            self.y += 1
        if self.action == 3:
            self.y -= 1
        self.position = (self.y, self.x)

    def undo_move(self):
        self.x = self.oldx
        self.y = self.oldy
        self.position = (self.y, self.x)

    def age_value(self):
        return self.current_age/self.max_age

    def memorize(self, old_state, new_state, action, reward, done, target_found):
        state_size = len(old_state)
        self.total_reward += reward
        self.memory.append([old_state.reshape(-1, state_size), new_state.reshape(-1, state_size), action, reward, done])
        if target_found:
            self.found_targets += 1
            self.current_age = 0

    def _random_action(self):
        return random.randint(0,len(self.action_space)-1)
