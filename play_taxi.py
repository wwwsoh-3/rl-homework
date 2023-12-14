import gymnasium as gym

from taxi import Agent

# 로드 하고 싶은 모델 파일 정의 
#model_path = './models/pytorch_1702476656.pt'
model_path = './models/taxi_dqn_1702558494.pt'


env = gym.make("Taxi-v3", render_mode="human").env
agent = Agent(env=env)        
agent.play( log_flag = True, sleep=0.2, max_steps=100,model_path = model_path)
env.close()
