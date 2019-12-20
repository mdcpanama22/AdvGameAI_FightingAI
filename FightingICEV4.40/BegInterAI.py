# from this import s
#
# import gym
# import gym_fightingice
# import numpy as np
#
# # List of hyper-parameters and constants
# DECAY_RATE = 0.99
# BUFFER_SIZE = 40000
# MINIBATCH_SIZE = 64
# TOT_FRAME = 3000000
# EPSILON_DECAY = 1000000
# MIN_OBSERVATION = 5000
# FINAL_EPSILON = 0.05
# INITIAL_EPSILON = 0.1
# NUM_ACTIONS = 6
# TAU = 0.01
# # Number of frames to throw into network
# NUM_FRAMES = 3
#
#
# class BrunchAI:
#     def __init__(self):
#         self.env = gym.make("FightingiceDataFrameskip-v0",
#                             java_env_path="C:/Users/mdcpa/FightingAI/FightingICEV4.40")
#         print([self.env.observation_space.low])
#         self.Q = np.zeros([self.env.observation_space.shape[0], self.env.action_space.n])
#         self.eta = .628
#         self.gma = .9
#         self.epis = 4
#         self.rev_list = []
#         self.First = True;
#
#     def testModel(self):
#         for i in range(self.epis):
#             if self.First:
#                 self.First = False
#             else:
#                 self.env = gym.make("FightingiceDataFrameskip-v0",
#                                     java_env_path="C:/Users/mdcpa/FightingAI/FightingICEV4.40")
#             rAll = 0
#             j = 0
#             rounds = 0
#             obs = self.env.beginMatch()
#             while j < 99:
#                 j += 1
#                 # chooose action from the Q Table
#                 print(obs)
#                 a = np.argmax(self.Q[obs, :] + np.random.randn(1, self.env.action_space.n) * (1. / (i + 1)))
#                 # get new reward
#                 s1, r, d, _ = self.env.step(a)
#                 self.Q[obs, a] = self.Q[obs, a] + self.eta * (r + self.gma(self.Q[s1, :]) - self.Q[obs, a])
#                 rAll += r
#                 obs = s1
#                 sumR = sumR + r
#                 if d:
#                     obs = self.env.reset()  # Reset rounds
#                     if rounds != 2:
#                         print("ROUND OVER")
#                         rounds += 1
#                     else:
#                         print("Match Over\n with results {0} and {1}".format(sumR))
#                         break
#             self.rev_list.append(rAll)
#         print("reward on all episodes " + str(sum(self.rev_list) / self.epis))
#         print("Final Values Q-table")
#         print(self.Q)
#
#
# BrunchAI().testModel()
import gym
import gym_fightingice

def run(env):
    rounds = 0
    obs, reward, done, info = env.step(env.action_space.sample())
    while 1:
        if done:
            print(reward)
            if rounds != 2:
                rounds += 1
            else:
                env.reset()
                return
        obs, reward, done, info = env.step(env.action_space.sample())


env = gym.make("FightingiceDisplayFrameskip-v0",
                                    java_env_path="C:/Users/mdcpa/FightingAI/FightingICEV4.40")
env.action_space.np_random.seed(123)
env.reset()  # Collects Round 1
run(env)