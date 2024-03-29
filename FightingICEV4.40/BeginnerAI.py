# -*- coding: utf-8 -*-
"""BoxingRame.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vxviaRTHL9eB6r1CEl2BX_oQuDPyxmrN
"""
#--disable-window
import os
import sys
from operator import itemgetter
from datetime import datetime

"""np.random.uniform(0, 1) -- [0, 1)"""
import gym
import gym_fightingice
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from multiprocessing import Process, freeze_support
import threading

import uuid


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, x, y, layer1_nodes, layer2_nodes):
        self.input = x
        self.weights1 = np.random.rand(len(x), layer1_nodes)
        self.weights2 = np.random.rand(layer1_nodes, layer2_nodes)
        self.weights3 = np.random.rand(layer2_nodes, len(y))
        self.output = np.zeros(len(y))

    def feedforward(self, x):
        self.input = x
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def assignWeights(self, w1, w2, w3):
        self.weights1 = w1
        self.weights2 = w2
        self.weights3 = w3


class BeginnerAI:
    def __init__(self):
        # NN variable
        self.env = None
        self.obs = None
        self.INPUT = 143
        self.IL = 6  # inner Layer
        self.OL = 4  # outer layer
        self.OUTPUT = 56

        self.pop = 8
        self.mRate = 62 / 100  # Rate for mutations
        self.pRate = 41 / 100  # Rate for adding a new child from scratch
        self.PRate = 37 / 100  # Rate for Parent mutation
        self.generations = 10
        self.n_indices = 15

        if self.pop <= 0:
            print("PLEASE CHOOSE A POPULATION GREATER THAN 1")
            quit()

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
        NN = self.generateParents()  # uses env to create the first step of the simulation pop n times
        highest = -1

        print("Starting with\n")
        print("Pop: ", self.pop, "mRate:", self.mRate, "n-indices:s", self.n_indices, "\n")
        print("NN-input:", self.INPUT, "NN-InnerLayer:", self.IL, "NN-OuterLayer:", self.OL, "NN-output:", self.OUTPUT,
              "\n")

        print("Start the Simulation\n")
        HighScore = []
        first = True

        self.env = gym.make("FightingiceDataFrameskip-v0",
                            java_env_path="C:/Users/mdcpa/PycharmProjects/AdvGameAI_FightingAI/FightingICEV4.40",
                            freq_restart_java= self.pop * self.generations) #Within gym.make, the program multiplies it by 3, this is the total amount of rounds the program is rendering
        self.obs = self.env.reset()
        for i in range(self.generations):
            NGeneration = []
            print("Generation ", i)
            countI = 0
            for CNN in NN:
                NGeneration.append(self.runOnce(CNN, countI))
                countI += 1
            NN = self.Fitness(
                NGeneration)  # Returns list of tuples (reward, NN, punishment) #PUNSHIMENT NOT ADDED ON THIS
            Contender = (NN[0][0], NN[0][1], NN[0][2])  # Highest fitness
            if self.pop < 5:
                print("CURRENT ", len(NN), " TOP CONTENDERS: ")
                for j in range(len(NN)):
                    print(NN[j][0], " ", NN[j][2])
            else:
                print("CURRENT 5 TOP CONTENDERS: ")
                for j in range(5):
                    print(NN[j][0], " ", NN[j][2])
            if Contender[0] > highest:
                print("GENERATION: ", i, "FITNESS: \nP1", Contender[0], " N", Contender[2])
                highest = Contender[0]
                mp1 = multiprocessing.Process(target=self.DataToCSV("Top_R{0}_G{1}".format(highest, i), Contender[1]))
                mp1.run()
            HighScore.append(Contender[0])
            NN = self.newGeneration(NN)
        print("Random")
        plt.subplot()
        plt.xlabel("GENERATION")
        plt.ylabel("Score")
        plt.plot(HighScore, 'ro')
        plt.show()

    def runOnce(self, NN, countI):
        rounds = 0
        sumR = 0 #Sum of Rewards
        sumG = 0 #Sum of Wins and losses; Win = 1, Loss = -1, Tie = 0
        first = True
        while 1:
            NN.feedforward(self.obs)
            choice = list(NN.output)
            action = self.findHighestList(choice)
            obs, reward, done, info = self.env.step(action)
            if done:

                sumG += reward
                print(reward)
                obs = self.env.reset()
                if rounds != 2:
                    rounds += 1
                    sumR += reward
                else:
                    return sumR, NN, sumG  # GREEDY, JUST GET TOTAL SUM OF THE 3 ROUNDS
            else:
                sumR = sumR + reward

    def newGeneration(self, NN):

        if self.pop <= 6:
            pSize = self.pop // 2
        else:
            pSize = 6
        #print("Setting up Parents, and Nsorted of size pSize")
        NParent = self.unFitness(NN,
                                 pSize)  # list of NN that survived with an updated NSorted: List of NN, and list of <NN>
        #print("Setting up the new Children of size ", self.pop, " - ", pSize)
        nChildren = self.generateChildren(NParent, self.pop - pSize,
                                          1)  # create a new list of NNchildren from NParents n-point
        # nChildren = self.generateChildrenCross(NParents, pop-5, 1)
        NParents = self.mutateParents(NParent)
        return NParents + nChildren

    def generateParents(self):
        NNg = []
        for i in range(self.pop):
            NNg.append(
                self.generateGenericChild())  # [0] * OUTPUT makes sure you only ouput the right amount of total moves
        return NNg

    def generateGenericChild(self):
        return NeuralNetwork(np.random.rand(self.INPUT), [0] * self.OUTPUT, self.IL, self.OL)

    def mutateParents(self, mP):
        for M in mP:
            # Turns Matrix into a list
            W1 = self.MatrixtoList(M.weights1)
            W2 = self.MatrixtoList(M.weights2)
            W3 = self.MatrixtoList(M.weights3)
            # mutate Parents
            W1 = self.mutate(W1)
            W2 = self.mutate(W2)
            W3 = self.mutate(W3)
            # Turn List back to Matrix
            W1 = self.ListToMatrix(W1, self.INPUT, self.IL)
            W2 = self.ListToMatrix(W2, self.IL, self.OL)
            W3 = self.ListToMatrix(W3, self.OL, self.OUTPUT)
            # Turn it back into an np.array
            W1 = np.array(W1)
            W2 = np.array(W2)
            W3 = np.array(W3)
            # Reasign weights to parents
            M.assignWeights(W1, W2, W3)
        return mP
        # This is only called when mutating Parent

    def mutate(self, NWAT):
        mutatedT = []
        # list of individual values
        for A in range(len(NWAT)):
            chanceRate = np.random.uniform(0, 1)
            if chanceRate <= self.PRate:  # PRate is chance a parent mutates
                mutatedT.append(np.random.uniform(0, 1))
            else:
                mutatedT.append(NWAT[A])
        return mutatedT

    # s is for strategy 1 - n-PointCrossover

    # Sort the generation by highest scoring, and least punishment
    def Fitness(self, NNR):
        # sorted(NNR, key=lambda e: (e[2]))
        # sorted(NNR, key=lambda y: (y[0]), reverse=True)
        #print("NNR:", NNR)
        NNR.sort(key=self.comparator, reverse=True)  # sort based on Reward/Fitness
        NNR = self.sortP1HighScore(
            NNR)  # Keep the same Reward order, but change them based on who has the lowest enemy score
        return NNR

    # Gets just the survivors that will be mutated and continued
    def unFitness(self, NNR, p):
        tempNN = []
        tempRNN = []
        for e in range(int(p)):
            tempNN.append(NNR[e][1])
        return tempNN

    # Returns the list of most probable move outputs
    def findHighestList(self, NN_Output):
        NN_Temp = []
        ind = 0
        for NNT in NN_Output:
            NN_Temp.append((NNT, ind))
            ind += 1
        res = sorted(NN_Temp, key=itemgetter(0), reverse=True)
        foundI = random.randint(0, 10)
        return res  # [foundI][1]

    def comparator(self, t2):
        return (t2[0])

    def comparator2(self, t2):
        return (t2[2])

    # We sort based on the highest score, but also as long as player2/computer don't have as mnuch
    # Takes the sorted NN Rewards based on the score of the other player
    # we want to see highest p1 score, versus low p2 score
    #THIS NEEDS TO HAVE A SORTED LIST BY FIRST POSITION AS THE INPUT
    def sortP1HighScore(self, NNR):
        value = -1
        listToSort = []
        NNR_F = []
        for N in NNR:
            if value == -1:
                value = N[0]
                listToSort.append(N)
            else:
                if value != N[0]: #this works because values are either equal to or greater than
                    listToSort.sort(key=self.comparator2)
                    for LTS in listToSort:
                        NNR_F.append(LTS)
                    listToSort = []
                    value = N[0]
                    listToSort.append(N)
                else:
                    listToSort.append(N)
        if len(listToSort) != 0:
            listToSort.sort(key=self.comparator2)
            for LTS in listToSort:
                NNR_F.append(LTS)
            listToSort = []
        return NNR_F

    # Mutating NOT parent
    def mutateNP(self, NWAT):
        mutatedT = []
        # turns it into a list of individual values
        for A in range(len(NWAT)):
            for B in range(len(NWAT[A])):
                chanceRate = np.random.uniform(0, 1)
                if chanceRate <= self.mRate:
                    mutatedT.append(np.random.uniform(0, 1))
                else:
                    mutatedT.append(NWAT[A][B])
        return mutatedT

    def MatrixtoList(self, WA):
        A = []
        for W in range(len(WA)):
            for WX in range(len(WA[W])):
                A.append(WA[W][WX])
        return A

        # Returns a list of indices to slice

    def nPointCrossoverIndices(self, n, WA):
        RJ = []
        if n > int(len(WA)):
            n = int(len(WA))
        maxN = int(len(WA) / n)
        if maxN == 1:
            for k in range(n):
                if k != n - 1:
                    RJ.append((k, k + 1))
                else:
                    RJ.append((k, int(len(WA))))
        else:
            nI = maxN  # n iterations
            rI = 0
            rJ = 0
            diff = 0

            for k in range(n):
                if k == 0:  # 0 - amount of n
                    rI = 0
                    rJ = random.randint(rI + 1, maxN)
                    diff = maxN - rJ  # This is so that it considers the values that were left behind if random value is below maxN

                elif k == n - 1:
                    rI = rJ
                    if rI >= int(len(WA)):
                        rI = int(len(WA))
                        rJ = rI
                    else:
                        rJ = int(len(WA))

                else:
                    nI = rJ + maxN + diff - 1
                    if (nI >= len(WA)):
                        nI = len(WA)
                    rI = rJ
                    rJ = random.randint(int(rI), int(nI))
                    diff = nI - rJ
                    if (rI == rJ and nI != len(WA)):
                        rJ = rJ + 1
                RJ.append((rI, rJ))
        return RJ

    # Returns two lists of values for children
    def nPointCrossover(self, I, WA, WB):
        NWATA = []
        NWATB = []
        jCount = 0
        for e in I:
            if jCount % 2 == 0:
                NWATA.append(WA[e[0]:e[1]])
                NWATB.append(WB[e[0]:e[1]])
            else:
                NWATA.append(WB[e[0]:e[1]])
                NWATB.append(WA[e[0]:e[1]])
            jCount = jCount + 1
        return self.mutateNP(NWATA), self.mutateNP(NWATB)

    def organizeW(self, W, row, col):
        count = 0
        FinalAB = []
        tempListA = []
        tempListB = []
        for ro in range(row):
            tB = []
            tA = []
            for c in range(col):
                tA.append(W[0][count])
                tB.append(W[1][count])
                count = count + 1
            tempListA.append(tA)
            tempListB.append(tB)
        FinalAB.append(tempListA)
        FinalAB.append(tempListB)
        return FinalAB

    def CrossOver(self, wa, wb):
        content = 0
        tempListA = []
        tempListB = []
        for ro in wa:
            coin = np.random.uniform(0, 1)
            if coin < 50:  # HEADS
                tempListA.append(wa[content])
                tempListB.append(wb[content])
            else:
                tempListA.append(wb[content])
                tempListB.append(wa[content])
            content = content + 1
        return tempListA, tempListB

    def ListToMatrix(self, WA, r, c):
        count = 0
        tempListA = []
        for ro in range(r):
            tA = []
            for co in range(c):
                tA.append(WA[count])
                count = count + 1
            tempListA.append(tA)
        return tempListA

    # return Matrix of Child A and Child B
    def organizeW(self, W, row, col):
        count = 0
        FinalAB = []
        tempListA = []
        tempListB = []
        for ro in range(row):
            tB = []
            tA = []
            for c in range(col):
                tA.append(W[0][count])
                tB.append(W[1][count])
                count = count + 1
            tempListA.append(tA)
            tempListB.append(tB)
        FinalAB.append(tempListA)
        FinalAB.append(tempListB)
        return FinalAB

    # returns the new weights for Child A and Child B
    # decides the strategy: TODO
    def createNewWeights(self, weightsA, weightsB, p):
        # Turns Matrix into a list
        WA = self.MatrixtoList(weightsA)
        WB = self.MatrixtoList(weightsB)

        # STEP 1 find list of indice
        Indices = self.nPointCrossoverIndices(self.n_indices, WA)

        # STEP 2 Perform slicing, and save it as a 1D list of values
        NWA = self.nPointCrossover(Indices, WA, WB)

        # STEP 3 make a list of lists using values, and their primary row by column values
        if p == 0:
            FWeights = self.organizeW(NWA, self.INPUT, self.IL)
        elif p == 1:
            FWeights = self.organizeW(NWA, self.IL, self.OL)
        else:
            FWeights = self.organizeW(NWA, self.OL, self.OUTPUT)

        # STEP 4 turn it back into an np.array
        NPAFinalA = np.array(FWeights[0])
        NPAFinalB = np.array(FWeights[1])

        return NPAFinalA, NPAFinalB

    def CrossoverChildren(self, W1, W2, W3):
        NNCA = self.generateGenericChild()
        NNCB = self.generateGenericChild()
        NNCA.assignWeights(W1[0], W2[0], W3[0])
        NNCB.assignWeights(W1[1], W2[1], W3[1])
        return NNCA, NNCB

    def generateChildren(self, N_p, x_g, s):
        NP = N_p
        newChildren = []
        if s == 1:
            for xg in range(int(x_g // 2)):
                rX = random.randint(0, len(N_p) - 1)
                rY = random.randint(0, len(N_p) - 1)
                while rX == rY:
                    rX = random.randint(0, int(len(N_p) - 1))
                    rY = random.randint(0, int(len(N_p) - 1))

                # New Matrix of Input to first layer <ChildA, ChildB>
                NW1 = self.createNewWeights(NP[rX].weights1, NP[rY].weights1, 0)
                # New Matrix of first layer to second layer <ChildA, ChildB>
                NW2 = self.createNewWeights(NP[rX].weights2, NP[rY].weights2, 1)
                # New Matrix of second layer to output <ChildA, ChildB>
                NW3 = self.createNewWeights(NP[rX].weights3, NP[rY].weights3, 2)

                # Returns NN childA and childB with new weights
                newC = self.CrossoverChildren(NW1, NW2, NW3)

                nRateC = np.random.uniform(0, 1)
                if nRateC <= self.pRate:
                    nD = np.random.uniform(0, 1)
                    newChildren.append(self.generateGenericChild())
                    if nD < 0.5:
                        newChildren.append(newC[1])
                    else:
                        newChildren.append(newC[0])
                else:
                    newChildren.append(newC[0])
                    newChildren.append(newC[1])
            return newChildren

    def createNewWeightsCross(self, weightsA, weightsB, p):
        # Turns Matrix into a list
        WA = self.MatrixtoList(weightsA)
        WB = self.MatrixtoList(weightsB)

        NWA, NWB = self.CrossOver(WA, WB)

        if p == 0:
            FWeights = (self.ListToMatrix(NWA, self.INPUT, self.IL), self.ListToMatrix(NWB, self.INPUT, self.IL))
        elif p == 1:
            FWeights = (self.ListToMatrix(NWA, self.IL, self.OL), self.ListToMatrix(NWB, self.IL, self.OL))
        else:
            FWeights = (self.ListToMatrix(NWA, self.OL, self.OUTPUT), self.ListToMatrix(NWB, self.OL, self.OUTPUT))

        # STEP 4 turn it back into an np.array
        NPAFinalA = np.array(FWeights[0])
        NPAFinalB = np.array(FWeights[1])

        return NPAFinalA, NPAFinalB

    # s is for strategy Crossover
    def generateChildrenCross(self, N_p, x_g, s):
        NP = N_p
        newChildren = []
        if s == 1:
            for xg in range(int(x_g // 2)):
                rX = random.randint(0, len(N_p) - 1)
                rY = random.randint(0, len(N_p) - 1)
                while rX == rY:
                    rX = random.randint(0, int(len(N_p) - 1))
                    rY = random.randint(0, int(len(N_p) - 1))

                # debugWeights(NP, rX, rY)

                # New Matrix of Input to first layer <ChildA, ChildB>
                NW1 = self.createNewWeightsCross(NP[rX].weights1, NP[rY].weights1, 0)
                # New Matrix of first layer to second layer <ChildA, ChildB>
                NW2 = self.createNewWeightsCross(NP[rX].weights2, NP[rY].weights2, 1)
                # New Matrix of second layer to output <ChildA, ChildB>
                NW3 = self.createNewWeightsCross(NP[rX].weights3, NP[rY].weights3, 2)

                # Returns NN childA and childB with new weights
                newC = self.CrossoverChildren(NW1, NW2, NW3)

                nRateC = np.random.uniform(0, 1)
                if nRateC <= self.pRate:
                    nD = np.random.uniform(0, 1)
                    newChildren.append(self.generateGenericChild())
                    if nD < 0.5:
                        newChildren.append(newC[1])
                    else:
                        newChildren.append(newC[0])
                else:
                    newChildren.append(newC[0])
                    newChildren.append(newC[1])
            return newChildren


def startBAI():
    BAI = BeginnerAI()
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
