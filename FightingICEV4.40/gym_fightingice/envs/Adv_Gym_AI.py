import random

import numpy as np
from py4j.java_gateway import get_field


class AdvGymAI(object):
    def __init__(self, gateway, pipe, frameskip=True):
        self.gateway = gateway
        self.pipe = pipe

        self.width = 96  # The width of the display to obtain
        self.height = 64  # The height of the display to obtain
        self.grayscale = True  # The display's color to obtain true for grayscale, false for RGB

        self.obs = None
        self.just_inited = True

        self._actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self.action_strs = self._actions.split(" ")

        self.pre_framedata = None

        self.frameskip = frameskip

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()

        self.player = player
        self.gameData = gameData

        return 0

    # please define this method when you use FightingICE version 3.20 or later
    #x is my hp y is opp and idk what z is
    def roundEnd(self, x, y, z):
        print("send round end to {}".format(self.pipe))
        Result = 0
        if x > y:
            print("Player 1 won")
            Result = 1
        elif y > x:
            print("Player 2 won")
            Result = -1
        else:
            print("Draw")
        p2_hp_now = self.frameData.getCharacter(False).getHp()
        p1_hp_now = self.frameData.getCharacter(True).getHp()
        self.pipe.send([self.obs, Result, True, None])
        self.just_inited = True
        # request = self.pipe.recv()
        # if request == "close":
        #     return
        self.obs = None

    # Please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def getInformation(self, frameData, isControl):
        self.pre_framedata = frameData if self.pre_framedata is None else self.frameData
        self.frameData = frameData
        self.isControl = isControl
        self.cc.setFrameData(self.frameData, self.player)
        if frameData.getEmptyFlag():
            return

    def input(self):
        return self.inputKey

    def gameEnd(self):
        pass

    def processStep(self):
        # print("waitting for step in {}".format(self.pipe.id))
        request = self.pipe.recv()
        # print("get step in {}".format(self.pipe))
        if len(request) == 2 and request[0] == "step":
            # s -- secret ex -- extra r -- real
            sAction = self.action_strs[request[1][random.randint(0, 10)][1]]  # request holds the sorted list of inputs
            # # Here I want to be able to split the inputs into different types of inputs between aggro, defensive, etc
            # # I am going to have to look into Q-Learning more
            # action = request[1]
            # exAction = ""
            # rAction = ""
            # distance = self.frameData.getDistanceX()
            #
            # my = self.frameData.getCharacter(self.player)
            # energy = my.getEnergy()
            # my_x = my.getX()
            # my_state = my.getState()
            #
            # opp = self.frameData.getCharacter(not self.player)
            # opp_x = opp.getX()
            # opp_state = opp.getState()
            #
            # xDifference = my_x - opp_x

            if self.cc.getSkillFlag():
                # If there is a previous "command" still in execution, then keep doing it
                self.inputKey = self.cc.getSkillKey()
                return
            # We empty the keys and cancel skill just in case
            self.inputKey.empty()
            self.cc.skillCancel()

            # # Following is the brain of the reflex agent. It determines distance to the enemy
            # # and the energy of our agent and then it performs an action
            # if (opp.getEnergy() >= 300) and (my.getHp() - opp.getHp() <= 300):
            #     # If the opp has 300 of energy, it is dangerous, so better jump!!
            #     # If the health difference is high we are dominating so we are fearless :)
            #     action = 31
            #     exAction = "_B B B"
            # elif not my_state.equals(self.gateway.jvm.enumerate.State.AIR) and not my_state.equals(
            #         self.gateway.jvm.enumerate.State.DOWN):
            #     # If not in air
            #     if distance > 150:
            #         # If its too far, then jump to get closer fast
            #         action = 31
            #         exAction = ""
            #     elif energy >= 300:
            #         # High energy projectile
            #         action = 44
            #         exAction = ""
            #     elif (distance > 100) and (energy >= 50):
            #         # Perform a slide kick
            #         action = 41
            #         exAction = ""
            #     elif opp_state.equals(self.gateway.jvm.enumerate.State.AIR):  # If enemy on Air
            #         # Perform a big punch
            #         action = 45
            #         exAction = ""
            #     elif distance > 100:
            #         # Perform a quick dash to get closer
            #         action = None
            #         exAction = "6 6 6"
            #     else:
            #         # Perform a kick in all other cases, introduces randomness
            #         action = None
            #         exAction = "B"
            # elif ((distance <= 150) and (my_state.equals(self.gateway.jvm.enumerate.State.AIR) or my_state.equals(
            #         self.gateway.jvm.enumerate.State.DOWN)) and (
            #               ((self.gameData.getStageWidth() - my_x) >= 200) or (xDifference > 0)) and (
            #               (my_x >= 200) or xDifference < 0)):
            #     # Conditions to handle game corners
            #     if energy >= 5:
            #         # Perform air down kick when in air
            #         action = 8
            #         exAction = ""
            #     else:
            #         # Perform a kick in all other cases, introduces randomness
            #         action = None
            #         exAction = "B"
            # else:
            #     # Perform a kick in all other cases, introduces randomness
            #     action = None
            #     exAction = "B"
            # if action is None:
            #     rAction = exAction
            # else:
            #     rAction = self.action_strs[action]
            #     if exAction != "":
            #         rAction += (" " + exAction)

            self.cc.commandCall(sAction)  # THE REAL COMMAND CALL
            if not self.frameskip:
                self.inputKey = self.cc.getSkillKey()
        if len(request) == 1 and request[0] == "end":
            self.pipe.send([self.frameData.getCharacter(self.player).getHP(),
                            self.frameData.getCharacter(not self.player).getHP()])

    def processing(self):
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingTime() <= 0:
            self.isGameJustStarted = True
            return

        if self.frameskip:
            if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                return
            if not self.isControl:
                return

            self.inputKey.empty()
            self.cc.skillCancel()

        # if just inited, should wait for first reset()
        if self.just_inited:
            request = self.pipe.recv()
            if request == "reset":
                self.just_inited = False
                self.obs = self.get_obs()
                self.pipe.send(self.obs)
            else:
                raise ValueError
        # if there is self.obs, do step() and return [obs, reward, done, info]
        else:
            self.obs = self.get_obs()
            self.reward = self.get_reward()
            self.pipe.send([self.obs, self.reward, False, None])
        self.processStep()

    def get_reward(self):
        try:
            if self.pre_framedata.getEmptyFlag() or self.frameData.getEmptyFlag():
                reward = 0
            else:
                p2_hp_pre = self.pre_framedata.getCharacter(False).getHp()
                p1_hp_pre = self.pre_framedata.getCharacter(True).getHp()
                p2_hp_now = self.frameData.getCharacter(False).getHp()
                p1_hp_now = self.frameData.getCharacter(True).getHp()
                if self.player:
                    reward = (p2_hp_pre - p2_hp_now) - (p1_hp_pre - p1_hp_now)
                else:
                    reward = (p1_hp_pre - p1_hp_now) - (p2_hp_pre - p2_hp_now)
        except:
            reward = 0
        return reward

    def get_obs(self):
        my = self.frameData.getCharacter(self.player)
        opp = self.frameData.getCharacter(not self.player)

        # my information
        myHp = abs(my.getHp() / 400)
        myEnergy = my.getEnergy() / 300
        myX = ((my.getLeft() + my.getRight()) / 2) / 960
        myY = ((my.getBottom() + my.getTop()) / 2) / 640
        mySpeedX = my.getSpeedX() / 15
        mySpeedY = my.getSpeedY() / 28
        myState = my.getAction().ordinal()
        myRemainingFrame = my.getRemainingFrame() / 70

        # opp information
        oppHp = abs(opp.getHp() / 400)
        oppEnergy = opp.getEnergy() / 300
        oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
        oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
        oppSpeedX = opp.getSpeedX() / 15
        oppSpeedY = opp.getSpeedY() / 28
        oppState = opp.getAction().ordinal()
        oppRemainingFrame = opp.getRemainingFrame() / 70

        # time information
        game_frame_num = self.frameData.getFramesNumber() / 3600

        observation = []

        # my information
        observation.append(myHp)
        observation.append(myEnergy)
        observation.append(myX)
        observation.append(myY)
        if mySpeedX < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(mySpeedX))
        if mySpeedY < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(mySpeedY))
        for i in range(56):
            if i == myState:
                observation.append(1)
            else:
                observation.append(0)
        observation.append(myRemainingFrame)

        # opp information
        observation.append(oppHp)
        observation.append(oppEnergy)
        observation.append(oppX)
        observation.append(oppY)
        if oppSpeedX < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedX))
        if oppSpeedY < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedY))
        for i in range(56):
            if i == oppState:
                observation.append(1)
            else:
                observation.append(0)
        observation.append(oppRemainingFrame)

        # time information
        observation.append(game_frame_num)

        myProjectiles = self.frameData.getProjectilesByP1()
        oppProjectiles = self.frameData.getProjectilesByP2()

        if len(myProjectiles) == 2:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            myHitDamage = myProjectiles[1].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
        elif len(myProjectiles) == 1:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        if len(oppProjectiles) == 2:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
        elif len(oppProjectiles) == 1:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        observation = np.array(observation, dtype=np.float32)
        observation = np.clip(observation, 0, 1)
        return observation

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
