import sys, importlib, math, random, time
from environment import Environment
from agent import Agent
from model import Model
from display import UI
from collections import deque
from statistics import mean
import numpy as np
import pygame

class ModelRunner:
    def __init__(self, config):
        random.seed(1)
        np.random.seed(1)
        self.env = Environment(
            {
                "move": config["rewards_move"],
                "obstacle": config["rewards_obstacle"],
                "backtrack": config["rewards_backtrack"],
                "target_found": config["rewards_target_found"]
            }, 
            config["agent_path_length"],
            config["state_grid_width"],
            config["env_handicap"]
            )
        self.config = config
        self.env.reset_grid()
        self.env.new_targets(1)
        self.agent = Agent(self.env.free_space(1)[0])

        # renderer
        self.UI = UI(self.env.grid.shape)
        self.render_step = self.UI.render_grid(self.env)
        self.render_step.send(None) # empty frame render, required by pygame

    def create_stats_holders(self):
        self.targets_found = deque(maxlen=self.config["aggregate_mean_over"])
        self.acc_rewards = deque(maxlen=self.config["aggregate_mean_over"])
        self.agent_ages = deque(maxlen=self.config["aggregate_mean_over"])
        self.targets_per_action = deque(maxlen=self.config["aggregate_mean_over"])

    def run(self):
        self.create_stats_holders()

        # a initial state value used for setting up the network (input shape)
        old_state = self.env.get_total_state(self.agent.position, self.agent.age_value())

        rewards = [self.config["rewards_move"], self.config["rewards_obstacle"], self.config["rewards_backtrack"], self.config["rewards_target_found"]]
        reward_range = max(rewards)-min(rewards)
        self.model = Model((1, len(old_state)), len(self.agent.action_space), self.config, reward_range)
        iterations = 0 
        while iterations < self.config.get("run_steps") + self.config.get("min_memory_size"):
            if (np.random.random() < self.model.epsilon) or not self.model.training_started:
                # random action taken
                self.agent.move()
            else:
                # action taken as suggested by the model.
                self.agent.move(self.model.suggest_move(old_state))
            state, reward, done, target_found, revert_move = self.env.evaluate(self.agent.position, self.agent.age_value())

            # if agent collides with a wall, undo the state transition.
            if revert_move:
                self.agent.undo_move()
                state = self.env.get_total_state(self.agent.position, self.agent.age_value())
            self.agent.memorize(old_state, state, self.agent.action, reward, done, target_found)
            old_state = state

            # rendering the current state.
            self.render_step.send({
                "env": self.env, 
                "agent": self.agent, 
                "done": done
            })
            self.UI.handle_input_events()
            if self.UI.manual_stop: break

            # iteration ended, store agent memory and train model
            if done:
                self.acc_rewards.append(self.agent.total_reward)
                self.targets_found.append(self.agent.found_targets)
                self.agent_ages.append(self.agent.total_age)
                self.targets_per_action.append(self.agent.targets_per_action)
                self.model.train_from_memory(
                    self.agent.memory,
                    mean(self.targets_found),
                    mean(self.acc_rewards),
                    mean(self.agent_ages),
                    mean(self.targets_per_action),
                    len(self.env.targets)
                )
                iterations +=1

                # rearrange targets and reset grid, agent and state
                self.env.reset_grid(self.model.epsilon)
                self.env.new_targets(math.ceil(self.model.epsilon * self.config["max_targets"]))
                self.agent = Agent(self.env.free_space(1)[0], iterations)
                old_state = self.env.get_total_state(self.agent.position, self.agent.age_value())