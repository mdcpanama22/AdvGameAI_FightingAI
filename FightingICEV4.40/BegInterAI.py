import datetime
import multiprocessing
import os
import sys

import gym_fightingice
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uuid

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import random
import tensorflow as tf

K.tensorflow_backend._get_available_gpus()

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class BegInterAI:
    def __init__(self):
        # NN variable
        self.state_size = None
        self.action_size = None
        self.AGENT = None
        self.env = None

        self.ID = uuid.uuid1()

    def checkDir(self, Dir):
        if not (os.path.exists(Dir)):
            os.mkdir(Dir)
        else:
            print("{0} already exists as a directory".format(Dir))

    def DataToCSV(self, name, NN):
        DataA = pd.DataFrame()
        DataB = pd.DataFrame()
        DataC = pd.DataFrame()
        DataO = pd.DataFrame()
        A = np.array(NN.weights1)
        DataA['WeightsA'] = A.flatten()
        DataB['WeightsB'] = NN.weights2.flatten()
        DataC['WeightsC'] = NN.weights3.flatten()
        DataO['output'] = NN.output.flatten()

        time = datetime.now()
        hour = time.strftime("%H_%M_%S")
        day = time.strftime("%d")
        strDir = "BestFighters"

        self.checkDir(strDir)
        strDir = "{0}/{1}".format(strDir, self.ID)
        self.checkDir(strDir)
        strDir = "{0}/{1}".format(strDir, "{0}_{1}_{2}".format(day, name, hour))
        self.checkDir(strDir)

        DataA.to_csv("{0}/WeightsA_data.csv".format(strDir), index=None, header=True)
        DataB.to_csv("{0}/WeightsB_data.csv".format(strDir), index=None, header=True)
        DataC.to_csv("{0}/WeightsC_data.csv".format(strDir), index=None, header=True)
        DataO.to_csv("{0}/Output_data.csv".format(strDir), index=None, header=True)

    # import sys
    # import subprocess
    #
    # procs = []
    # for i in range(86):
    #     proc = subprocess.Popen([sys.executable, 'task.py', '{}in.csv'.format(i), '{}out.csv'.format(i)])
    #     procs.append(proc)
    #
    # for proc in procs:
    #     proc.wait()
    # Grab best top half
    def MainBI(self):
        self.env = gym.make("FightingiceDisplayFrameskip-v0",
                                    java_env_path="C:/Users/mdcpa/PycharmProjects/AdvGameAI_FightingAI/FightingICEV4.40")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.AGENT = DQNAgent(self.state_size, self.action_size)

        results = []
        rounds = 0
        for i in range(2):
            state = self.env.reset()
            state = np.reshape(state, [64, self.state_size])
            currentC = 0
            while 1:
                action = self.AGENT.act(state)

                n_state, reward, done, _ = self.env.step(action)
                n_state = np.reshape(n_state, [64, self.state_size])

                state = n_state

                if done:
                    rounds += 1
                    if rounds == 2:
                        print("Episode: {}/{}, score: {}".format(i, 10, currentC))
                        results.append(currentC)
                        break
                currentC += 1
            #self.AGENT.replay(6)
        plt.plot(results)
                #self.runOnce()



def startBAI():
    BAI = BegInterAI()
    BAI.MainBI()


# if __name__ == '__main__':
#     import multiprocessing
#     from multiprocessing import Process
#     import threading
#
#     mp1 = multiprocessing.Process(target=startBAI)
#     mp2 = multiprocessing.Process(target=startBAI)
#
#     mp1.start()
#     mp2.start()
if __name__ == '__main__':
    startBAI()
    print("worked?")
    sys.exit()