import gymnasium as gym
from taxi import Agent

env = gym.make("Taxi-v3").env
agent = Agent(env=env)
agent.learn()
env.close()
