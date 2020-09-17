import random
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

class Environment:
    targets = []
    grid_values_range = 5
    grid = []
    past_positions = []
    

    def __init__(self, rewards, agent_path_length, state_grid_width, handicap):
        np.random.seed(1)
        random.seed(1)
        self.observed_tiles = int(state_grid_width / 2)
        self.handicap = handicap
        self.agent_path_length = agent_path_length
        self.rewards = rewards
        self.reset_grid()

    def reset_grid(self, epsilon=1):
        self.grid = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,3,3,3,3,0,3,3,3,0,0,3,0,3,3,3,3,3,0,0],
            [0,3,3,0,0,0,3,3,3,0,3,3,0,3,3,0,0,0,0,0],
            [0,3,3,3,3,3,3,3,3,0,3,3,3,3,3,3,3,0,0,0],
            [0,3,3,0,3,3,0,3,3,3,3,0,0,0,0,3,3,3,3,0],
            [0,3,0,0,0,3,0,0,0,0,3,0,3,0,0,0,3,3,3,0],
            [0,3,3,0,3,3,3,3,0,0,3,0,3,3,3,3,3,3,3,0],
            [0,3,3,3,3,0,3,3,3,3,3,3,3,0,3,3,0,3,3,0],
            [0,3,3,3,3,0,3,0,0,0,0,0,3,3,3,3,3,3,0,0],
            [0,3,3,3,0,0,0,0,0,0,3,3,3,3,3,3,0,0,0,0],
            [0,3,3,0,0,3,3,3,0,0,3,0,0,3,3,3,3,3,0,0],
            [0,3,3,0,0,3,3,3,0,0,3,0,0,0,0,3,3,3,0,0],
            [0,3,3,3,3,3,3,3,3,3,3,3,3,3,0,3,3,3,3,0],
            [0,3,0,0,3,0,0,3,0,0,0,0,3,3,0,0,0,0,3,0],
            [0,3,0,0,3,3,3,3,3,3,3,0,3,3,0,0,3,0,3,0],
            [0,3,3,3,3,0,0,3,0,3,3,3,3,3,3,3,3,3,3,0],
            [0,3,0,0,3,3,0,3,3,3,0,0,0,0,0,3,0,3,3,0],
            [0,3,0,0,0,3,0,3,0,3,0,3,3,3,0,3,0,0,3,0],
            [0,3,3,3,3,3,3,3,0,0,0,3,0,3,3,3,3,3,3,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ])
        self.grid[self.grid != 0] = 3
        self.grid = np.fliplr(self.grid.transpose())
        for i, row in enumerate(self.grid):
            if i > 0 and i < len(self.grid)-1:
                for j, col in enumerate(row):
                    if j > 0 and j < len(self.grid[i])-1:
                        if col == 0 and np.random.random() < epsilon * self.handicap:
                            self.grid[i][j] = 3

    def free_space(self, n_positions):
        free_space = []
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col == 3:
                    free_space.append((i, j))
        return list(random.sample(free_space, n_positions))

    def new_targets(self, n_targets):
        self.targets = []
        for t in self.free_space(n_targets):
            self.targets.append(t)
            self.grid[t] = 4

    def update_grid(self):
        self.grid[self.grid > 1] = 3
        for t in self.targets:
            self.grid[t] = 4

    def get_total_state(self, p, age):
        state = self._get_grid_state(p)
        distance_matrix = self._distance_to_target(p)
        s = np.concatenate((state/self.grid_values_range, distance_matrix), axis=0)
        s = np.concatenate((s, np.array([age])), axis=0)
        s = np.concatenate((s, self._targets_in_sight(p)), axis=0)
        return s

    def evaluate(self, p, age):
        self.update_grid()
        self._store_position_on_grid(p)
        target_found = False
        done = False
        revert_move = False
        state = self.get_total_state(p, age)
        total_reward = self.rewards["move"]
        if self.grid[p] == 0:
            total_reward = self.rewards["obstacle"] 
            revert_move = True 
            done = True
            self.past_positions = []
        if self.grid[p] == 1:
            total_reward += self.rewards["backtrack"] * 2
        if self.grid[p] == 2:
            self.grid[p] = 1
            total_reward += self.rewards["backtrack"]
        if p in self.targets:
            self.targets[self.targets.index(p)] = self.free_space(1)[0]
            target_found = True
            total_reward = self.rewards["target_found"]
            self.grid[self.grid > 0] = 3
            self.past_positions = []
        return state, total_reward, done, target_found, revert_move

    def _targets_in_sight(self, p):
        directions = [0,0,0,0]
        for i in range(0, p[1]):
            if self.grid[p[0]][i] == 4: directions[1] = 1
            if self.grid[p[0]][i] == 0: directions[1] = 0
        for i in reversed(range(p[1]+1, len(self.grid[p[0]]))):
            if self.grid[p[0]][i] == 4: directions[0] = 1
            if self.grid[p[0]][i] == 0: directions[0] = 0
        for i in range(0, p[0]):
            if self.grid[i][p[1]] == 4: directions[3] = 1
            if self.grid[i][p[1]] == 0: directions[3] = 0
        for i in reversed(range(p[0]+1, len(self.grid))):
            if self.grid[i][p[1]] == 4: directions[2] = 1
            if self.grid[i][p[1]] == 0: directions[2] = 0
        return np.array(directions)

    def _store_position_on_grid(self,p):
        self.past_positions = self.past_positions[-self.agent_path_length:]
        for prev_pos in self.past_positions:
            if self.grid[prev_pos] == 3:
                self.grid[prev_pos] = 2
        if self.grid[p] > 1:
            self.past_positions.append(p)

    def _get_grid_state(self, p):
        matrix = []
        for i in range(-self.observed_tiles,self.observed_tiles+1):
            padding_left = []
            padding_right = []
            if p[0] + i < 0 or p[0] + i > len(self.grid) -1:
                matrix.append([1 for n in range(self.observed_tiles*2+1)])
            else:
                if p[1]-self.observed_tiles < 0:
                    padding_left = [1 for j in range(p[1]-self.observed_tiles,0)]
                    start_x = 0
                else:
                    start_x = p[1]-self.observed_tiles
                if p[1]+self.observed_tiles > len(self.grid[0])-1:
                    padding_right = [1 for j in range(len(self.grid[1])-1,p[1]+self.observed_tiles)]
                matrix.append(
                    padding_left +
                    list(self.grid[p[0] + i][start_x:p[1]+self.observed_tiles+1]) +
                    padding_right
                )
        #pp.pprint(matrix) # uncomment to log state (can be very usefull for debugging)
        return np.array(matrix).flatten()

    def _distance_to_target(self, p):
        relative_targets = []
        distance_matrix = []
        for t in self.targets:
            relative_targets.append([(p[1]-t[1]) / len(self.grid[0]), (p[0]-t[0]) / len(self.grid)])
            distance_matrix.append((abs(p[1]-t[1]) + abs(p[0]-t[0])) / (len(self.grid) + len(self.grid[0])))
        return np.array(relative_targets[np.argmin(distance_matrix)])