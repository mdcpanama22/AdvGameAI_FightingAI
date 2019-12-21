import gym
import gym_fightingice

env = gym.make("FightingiceDataFrameskip-v0", java_env_path=".", port=4242)

# observation = env.reset (p2 = 'MyFighter') # You can specify your opponent's AI name (Java class name) in p2.
Obs = env.reset(p2='AIFighter')
done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    if reward != 0:
        print("Obs", obs, "Reward", reward, "DONE", done, "INFO", info)
print("DONE")
env.close()
exit()
