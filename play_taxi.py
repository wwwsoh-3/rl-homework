import gymnasium as gym

from taxi import Agent

env = gym.make("Taxi-v3", render_mode="human").env
agent = Agent(env=env)        
agent.play( log_flag = True, sleep=0.2, max_steps=100,model_path = './models/pytorch_1702476656.pt')
env.close()
