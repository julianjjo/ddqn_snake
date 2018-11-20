import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gym_snake
import gym
import random
import time
import numpy as np
from keras.models import model_from_json

if __name__ == '__main__':
    episodes = 200
    env = gym.make('snake-v0')
    state_size = 400
    actions_size = 4
    json_file = open('model_ddqn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model_ddqn.h5")
    print("Loaded model from disk")
    for i_episode in range(episodes):
        state = env.reset()
        state = state.flatten()
        score = 0
        done = False
        max_steps = 50
        steps = 0
        while not done:
            state = np.asarray([state])
            q_values = model.predict(state)[0]
            print(q_values)
            action = np.argmax(q_values)
            next_state, reward, done, info = env.step(action)
            if reward == 1:
                max_steps += 50
            if steps == max_steps:
                done = True
                reward = -1
            env.render()
            time.sleep(0.1)
            next_state = next_state.flatten()
            score += reward
            state = next_state
            if done:
                print("Episode finished {} with score {}".format(i_episode, score))
                break
            steps += 1
    env.close()
