import os
import platform
import random
import subprocess
import sys
import time
from multiprocessing import Pipe
from threading import Thread

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from py4j.java_gateway import (CallbackServerParameters, GatewayParameters,
                               JavaGateway, get_field)

import gym_fightingice
from gym_fightingice.envs.gym_ai import GymAI
from gym_fightingice.envs.Machete import Machete
from gym_fightingice.envs.Adv_Gym_AI import AdvGymAI


def game_thread(env):
    try:
        env.game_started = True
        env.manager.runGame(env.game_to_start)
    except:
        env.game_started = False
        print("Please IGNORE the Exception above because of restart java game")


class FightingiceEnv_Data_Frameskip(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        #  freq_restart_java=3, env_config=None, java_env_path=None, port=None, auto_start_up=False

        self.freq_restart_java = 3
        self.java_env_path = os.getcwd()

        if "java_env_path" in kwargs.keys():
            self.java_env_path = kwargs["java_env_path"]
        if "freq_restart_java" in kwargs.keys():
            self.freq_restart_java = kwargs["freq_restart_java"]
        if "port" in kwargs.keys():
            self.port = kwargs["port"]
        else:
            try:
                # https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
                # Had to do this because the default  port_for doesn't work
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
                s.close()
                self.port = port
                # import port_for
                # self.port = port_for.select_random()  # select one random port for java env
            except:
                raise ImportError(
                    "Pass port=[your_port] when make env, or install port_for to set startup port automatically, maybe pip install port_for can help")

        _actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self.action_strs = _actions.split(" ")

        self.observation_space = spaces.Box(low=0, high=1, shape=(143,))
        self.action_space = spaces.Discrete(len(self.action_strs))

        os_name = platform.system()
        if os_name.startswith("Linux"):
            self.system_name = "linux"
        elif os_name.startswith("Darwin"):
            self.system_name = "macos"
        else:
            self.system_name = "windows"

        if self.system_name == "linux":
            # first check java can be run, can only be used on Linux
            java_version = subprocess.check_output(
                'java -version 2>&1 | awk -F[\\\"_] \'NR==1{print $2}\'', shell=True)
            if java_version == b"\n":
                raise ModuleNotFoundError("Java is not installed")
        else:
            print("Please make sure you can run java if you see some error")

        # second check if FightingIce is installed correct
        start_jar_path = os.path.join(self.java_env_path, "FightingICE.jar")
        start_data_path = os.path.join(self.java_env_path, "data")
        start_lib_path = os.path.join(self.java_env_path, "lib")
        lwjgl_path = os.path.join(start_lib_path, "lwjgl", "*")
        lib_path = os.path.join(start_lib_path, "*")
        start_system_lib_path = os.path.join(
            self.java_env_path, "lib", "natives", self.system_name)
        natives_path = os.path.join(start_system_lib_path, "*")
        if os.path.exists(start_jar_path) and os.path.exists(start_data_path) and os.path.exists(
                start_lib_path) and os.path.exists(start_system_lib_path):
            pass
        else:
            error_message = "FightingICE is not installed in your script launched path {}, set path when make() or start script in FightingICE path".format(
                self.java_env_path)
            raise FileExistsError(error_message)

        self.java_ai_path = os.path.join(self.java_env_path, "data", "ai")
        ai_path = os.path.join(self.java_ai_path, "*")
        if self.system_name == "windows":
            self.start_up_str = "{};{};{};{};{}".format(
                start_jar_path, lwjgl_path, natives_path, lib_path, ai_path)
            self.need_set_memory_when_start = True
        else:
            self.start_up_str = "{}:{}:{}:{}:{}".format(
                start_jar_path, lwjgl_path, natives_path, lib_path, ai_path)
            self.need_set_memory_when_start = False

        self.game_started = False
        self.round_num = 0

    def _start_java_game(self):
        # start game
        print("Start java env in {} and port {}".format(
            self.java_env_path, self.port))
        devnull = open(os.devnull, 'w')

        if self.system_name == "windows":
            # -Xms1024m -Xmx1024m we need set this in windows
            self.java_env = subprocess.Popen(
                ["java", "-Xms2048m", "-Xmx2048m", "-cp", self.start_up_str, "Main", "--port", str(self.port), "--py4j",
                 "--fastmode",
                 "--grey-bg", "--inverted-player", "1", "--mute", "--limithp", "200", "200"], stdout=devnull,
                stderr=devnull)
        elif self.system_name == "linux":
            self.java_env = subprocess.Popen(
                ["java", "-cp", self.start_up_str, "Main", "--port", str(self.port), "--py4j", "--fastmode",
                 "--grey-bg", "--inverted-player", "1", "--mute", "--limithp", "400", "400"], stdout=devnull,
                stderr=devnull)
        elif self.system_name == "macos":
            self.java_env = subprocess.Popen(
                ["java", "-XstartOnFirstThread", "-cp", self.start_up_str, "Main", "--port", str(self.port), "--py4j",
                 "--fastmode",
                 "--grey-bg", "--inverted-player", "1", "--mute", "--limithp", "400", "400"], stdout=devnull,
                stderr=devnull)
        # self.java_env = subprocess.Popen(["java", "-cp", "/home/myt/gym-fightingice/gym_fightingice/FightingICE.jar:/home/myt/gym-fightingice/gym_fightingice/lib/lwjgl/*:/home/myt/gym-fightingice/gym_fightingice/lib/natives/linux/*:/home/myt/gym-fightingice/gym_fightingice/lib/*", "Main", "--port", str(self.free_port), "--py4j", "--c1", "ZEN", "--c2", "ZEN","--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"])
        # sleep 3s for java starting, if your machine is slow, make it longer
        time.sleep(3)  # originally 3 but to be safe i put 10

    def _start_gateway(self, p2=Machete):
        # auto select callback server port and reset it in java env
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(
            port=self.port), callback_server_parameters=CallbackServerParameters(port=0))
        python_port = self.gateway.get_callback_server().get_listening_port()
        self.gateway.java_gateway_server.resetCallbackClient(
            self.gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
        self.manager = self.gateway.entry_point

        # create pipe between gym_env_api and python_ai for java env
        server, client = Pipe()
        self.pipe = server
        # THIS IS WHERE THE PROGRAM CHOSES THE AI
        # Found in from gym_fightingice/envs/Adv_Gym_AI import AdvGymAI
        self.p1 = AdvGymAI(self.gateway, client, True)
        self.manager.registerAI(self.p1.__class__.__name__, self.p1)

        if isinstance(p2, str):
            # p2 is a java class name
            self.p2 = p2
            self.game_to_start = self.manager.createGame(
                "ZEN", "ZEN", self.p1.__class__.__name__, self.p2, self.freq_restart_java)
        else:
            # p2 is a python class
            self.p2 = p2(self.gateway)
            self.manager.registerAI(self.p2.__class__.__name__, self.p2)
            self.game_to_start = self.manager.createGame(
                "ZEN", "ZEN", self.p1.__class__.__name__, self.p2.__class__.__name__, self.freq_restart_java)
        self.game = Thread(target=game_thread,
                           name="game_thread", args=(self,))
        self.game.start()

        self.game_started = True
        self.round_num = 0

    def _close_gateway(self):
        self.gateway.shutdown_callback_server()
        self.gateway.shutdown()
        self.gateway.close_callback_server()
        self.gateway.close()
        del self.gateway

    def _close_java_game(self):
        self.java_env.kill()
        del self.java_env
        self.pipe.close()
        del self.pipe
        self.game_started = False

    def beginMatch(self, p2=Machete):
        self._start_java_game()
        self._start_gateway(p2)

        self.pipe.send("reset")
        self.round_num += 1
        obs = self.pipe.recv()
        return obs

    def endGame(self):
        try:
            self.game_started = False
            self._close_java_game()
            self._close_gateway()
            self.game.join()
            print("Unavoidable Exception")
            # self._start_java_game()
        except:
            raise SystemExit("Can not restart game")

    def reset(self, p2=Machete):
        # to provide crash, restart java game in some freq
        if self.round_num == self.freq_restart_java:
            self.endGame()
            return -1
        #     self._start_gateway(p2)
        # just reset
        self.pipe.send("reset")
        self.round_num += 1
        obs = self.pipe.recv()
        return obs

    def step(self, action):
        if self.game_started:
            self.pipe.send(["step", action])
            new_obs, reward, done, info = self.pipe.recv()
            return new_obs, reward, done, {}

    # def reset(self, p2=Machete):
    #     # start java game if game is not started
    #     if self.game_started is False:
    #         try:
    #             self._close_gateway()
    #             self._close_java_game()
    #         except:
    #             pass
    #         self._start_java_game()
    #         self._start_gateway(p2)
    #
    #     # to provide crash, restart java game in some freq
    #     if self.round_num == self.freq_restart_java * 3:  # 3 is for round in one game
    #         try:
    #             self._close_gateway()
    #             self._close_java_game()
    #             self._start_java_game()
    #         except:
    #             raise SystemExit("Can not restart game")
    #         self._start_gateway(p2)
    #
    #     # just reset is anything ok
    #     self.pipe.send("reset")
    #     self.round_num += 1
    #     obs = self.pipe.recv()
    #     return obs
    #
    # def step(self, action):
    #     # check if game is running, if not try restart
    #     # when restart, dict will contain crash info, agent should do something, it is a BUG in this version
    #     if self.game_started is False:
    #         dict = {}
    #         dict["pre_game_crashed"] = True
    #         return self.reset(), 0, None, dict
    #
    #     self.pipe.send(["step", action])
    #     new_obs, reward, done, info = self.pipe.recv()
    #     return new_obs, reward, done, {}


    def render(self, mode='human'):
        # no need
        pass

    def close(self):
        self._close_java_game()


if __name__ == "__main__":
    env = gym.make("FightingiceDataFrameskip-v0", java_env_path="/home/usen_name/FTG4.40")

    # env = FightingiceEnv_Data_Frameskip()

    # while True:
    #     obs = env.reset()
    #     done = False

    #     while not done:
    #         new_obs, reward, done, _ = env.step(random.randint(0, 10))

    # print("finish")
