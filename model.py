import random
from tensorflow.keras import Sequential
import numpy as np
import time
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tb import ModifiedTensorBoard
from collections import deque
import json
from tensorflow.keras import regularizers

class Model:
    def __init__(self, input_shape, action_space, args, reward_range):
        tf.random.set_seed(1)
        random.seed(1)
        np.random.seed(1)
        self.name = args["name"]
        self.discount = args["discount"]
        self.min_learning_rate = args["min_learning_rate"]
        self.learning_rate = args["learning_rate"]
        self.learning_rate_decay = args["learning_rate_decay"]
        self.min_epsilon = args["min_epsilon"]
        self.epsilon = args["epsilon"]
        self.epsilon_decay = args["epsilon_decay"]
        self.replay_memory = deque(maxlen=args["total_memory"])
        self.min_memory_size = args["min_memory_size"]
        self.batch_size = args["batch_size"]
        self.target_model_replace_at = args["target_model_replace_at"]
        self.recycle_memory_threshold = reward_range / 2
        self.target_model_replace_counter = 0
        self.model_generation = 0
        self.training_started = False
        self.tensorboard_callback = ModifiedTensorBoard(self.name, log_dir="logs/{}".format(self.name))
        print("creating model", self.name)
        self.model = self._create_dqn(input_shape, action_space, args["hidden_layers"], args["loss_function"])
        self.target_model = self._create_dqn(input_shape, action_space, args["hidden_layers"], args["loss_function"])
        self.target_model.set_weights(self.model.get_weights())
        self.new_memory_added = 0

    def suggest_move(self, state):
        return np.argmax(self.model.predict(state.reshape(-1, len(state))))

    def train_from_memory(self, agent_memory, avg_targets_found, mean_reward, mean_age, mean_targets_per_action, targets_in_env):
        self.replay_memory += agent_memory
        self.new_memory_added += len(agent_memory)
        self.tensorboard_callback.update_stats(
            mean_age=mean_age, 
            mean_reward_per_step=mean_reward/mean_age,
            targets_found=avg_targets_found, 
            model_generation=self.model_generation, 
            epsilon=self.epsilon, 
            discount=self.discount, 
            learning_rate=round(self.learning_rate, 5), 
            memory=len(self.replay_memory), 
            update_counter=self.target_model_replace_counter,
            targets_per_action=mean_targets_per_action,
            tagets_in_env=targets_in_env
        )
        if len(self.replay_memory) < self.min_memory_size or self.new_memory_added < self.batch_size:
            return
        else:
            self.new_memory_added = 0
        if not self.training_started: 
            print("training started")
            self.training_started = True
        self._fit_model(random.sample(self.replay_memory, self.batch_size))
        if self.target_model_replace_counter >= self.target_model_replace_at:
            print("replacing target model")
            self.target_model_replace_counter = 0
            self.target_model.set_weights(self.model.get_weights())
            self.model_generation += 1 
            print("saving model summary")
            self._save()
            if self.learning_rate > self.min_learning_rate:
                self.learning_rate *= self.learning_rate_decay
                K.set_value(self.model.optimizer.learning_rate, self.learning_rate)
        self.target_model_replace_counter += 1
        print("batch processed") 
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

    def _fit_model(self, batches):
        for old_state, new_state, action, reward, done in batches:   
            if done:
                target = reward
            else:
                target = reward + self.discount * np.max(self.target_model.predict(new_state))
            target_vec = self.model.predict(old_state)[0]
            target_vec[action] = target
            if abs(target_vec[action] - target) > self.recycle_memory_threshold:
                print("Re-adding memory. Prediction error: ", abs(target_vec[action] - target))
                self.total_memory.append([old_state, new_state, action, reward, done])
            self.model.fit(
                old_state, 
                target_vec.reshape(-1, 4), 
                epochs=1, 
                verbose=0, 
                callbacks=[self.tensorboard_callback]
            )

    def _save(self):
        self.model.save("models/{}-{}".format(self.name, self.tensorboard_callback.step))

    def _create_dqn(self, input_shape, action_space, hidden_layers, loss_function):
        model = Sequential()
        model.add(InputLayer(batch_input_shape=input_shape))
        for l in hidden_layers.split(","):
            model.add(Dense(int(l), activation='relu'))
        model.add(Dense(action_space, activation='linear'))
        model.compile(
            loss=loss_function, 
            optimizer=Adam(lr=self.learning_rate), 
            metrics=['accuracy']
        )
        return model


