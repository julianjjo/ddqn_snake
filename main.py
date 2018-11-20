import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gym_snake
import gym
import random
import numpy as np
from collections import deque
import scipy.misc
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.models import model_from_json


class DDQN(object):
    """docstring for DDQN."""
    def __init__(self, state_size=2, action_size=3, episodios=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.is_load = True
        self.lengthMemory = 64
        self.discountFactor = 0.99
        self.minLearningRate = 0.01
        self.maxLearningRate = 1.0
        self.learningRate = np.linspace(self.maxLearningRate, self.minLearningRate, episodios)
        self.discountRate = 0.1
        self.batch = deque(maxlen=2000)
        self.positive = deque(maxlen=300)
        self.ddqn = self.get_model_ddqn()
        self.target_ddqn = self.get_model_target_ddqn()
        self.update_target_model()
        if self.is_load:
            self.load_models()

    def get_model_ddqn(self):
        model = Sequential()
        model.add(Dense(34, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(34, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))

        model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy'])
        return model

    def get_model_target_ddqn(self):
        model = Sequential()
        model.add(Dense(34, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(34, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))

        model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy'])
        return model

    def get_action(self, state, episode):
        randomFloat = random.random()
        if np.random.rand() < self.learningRate[episode]:
            return random.randrange(self.action_size)
        else:
            state = np.asarray([state])
            q_values = self.ddqn.predict(state)[0]
            action = np.argmax(q_values)
            return action

    def train(self):
        train_batch_x = []
        train_batch_y = []
        len_mini_batch = min(len(self.batch), self.lengthMemory)
        mini_batch = random.sample(self.batch, len_mini_batch)
        len_positive_batch = min(len(self.positive), 10)
        positive_mini_batch = random.sample(self.batch, len_positive_batch)
        mini_batch.extend(positive_mini_batch)
        for step in mini_batch:
            state, action, reward, next_state, done  = self.get_values_step(step)
            train_batch_x.append(state)
            next_state = np.asarray([next_state])
            state = np.asarray([state])
            target_q_values = self.target_ddqn.predict(next_state)[0]
            q_values = self.ddqn.predict(state)[0]
            next_q_values = self.ddqn.predict(next_state)[0]
            a = np.argmax(next_q_values)

            if done is True:
                q_values[action] = reward;
            else:
                q_values[action] = reward + self.discountFactor * target_q_values[a];
            # q_values = np.asarray(q_values)
            train_batch_y.append(q_values)
        train_batch_x = np.asarray(train_batch_x)
        train_batch_y = np.asarray(train_batch_y)
        batch_size = len(train_batch_y)
        self.ddqn.train_on_batch(train_batch_x, train_batch_y)


    def get_values_step(self, step):
        state = step[0]
        action = step[1]
        reward = step[2]
        next_state = step[3]
        done = step[4]
        return state, action, reward, next_state, done

    def add_to_batch(self, state, action, reward, next_state, done):
        step = []
        step.append(state)
        step.append(action)
        step.append(reward)
        step.append(next_state)
        step.append(done)
        if reward == 100:
            self.positive.append(step)
        self.batch.append(step)

    def reset_batch(self):
        self.batch = []

    def load_models(self):
        self.ddqn.load_weights("model_ddqn.h5")
        print("Load model ddqn to disk")
        self.ddqn.load_weights("model_target_ddqn.h5")
        print("Load model target_ddqn to disk")

    def save_models(self):
        model_json = self.ddqn.to_json()
        with open("model_ddqn.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.ddqn.save_weights("model_ddqn.h5")
        print("Saved model ddqn to disk")

        model_json = self.target_ddqn.to_json()
        with open("model_target_ddqn.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.ddqn.save_weights("model_target_ddqn.h5")
        print("Saved model target ddqn to disk")

    def update_target_model(self):
        self.target_ddqn.set_weights(self.ddqn.get_weights())


if __name__ == '__main__':
    episodes = 50000
    env = gym.make('snake-v0')
    state_size = 400
    actions_size = 4
    ddqn = DDQN(state_size, actions_size, episodes)
    for i_episode in range(episodes):
        state = env.reset()
        state = state.flatten()
        score = 0
        done = False
        max_steps = 50
        steps = 0
        while not done:
            action = ddqn.get_action(state, i_episode)
            next_state, reward, done, info = env.step(action)
            if reward == 1:
                max_steps += 50
            if steps == max_steps:
                done = True
                reward = -1
            if reward == 1:
                reward = 100
            # env.render()
            next_state = next_state.flatten()
            ddqn.add_to_batch(state, action, reward, next_state, done)
            ddqn.train()
            score += reward
            state = next_state
            if done:
                print("Episode finished {} with score {}".format(i_episode, score))
                ddqn.update_target_model()
                break
            steps += 1
        if i_episode % 100 == 0:
            ddqn.save_models()
    env.close()
