import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import os
import time
from  pathlib import Path
import numpy as np

from collections import namedtuple
import gymnasium as gym


from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple('Transition',('state', 'action', 'state_prime', 'reward', 'done'))

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size= max_size
        self.pos=0
        self.memory=[]
    def push (self, *args):
        if len(self.memory) < self.max_size:
            self.memory.append(None)

        self.memory[self.pos]=Transition(*args)
        self.pos= (self.pos+1) % self.max_size


    def sample(self, batch_size):
        index = np.random.default_rng().choice(np.arange(len(self.memory)), batch_size, replace=False)
        res = []
        for i in index:
            res.append(self.memory[i])
        return res
        
        
    def get_memory_size(self):
        return len(self.memory)


class DQN(nn.Module): 
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(500, 4)
        self.l1 = nn.Linear(4, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, outputs)
        
    def forward(self, x):
        x = F.relu(self.l1(self.emb(x)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Agent(): 
    def __init__(self, env):
        self.env = env
        self.model_path = './models'
        self.writer = SummaryWriter('logs/tb_taxi')

        # hyper parameter 
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_episode = 5000 #5000
        self.train_step = 1000000
        self.warm_episode = 10
        self.save_freq = 1000
        self.max_step_per_epi = 200
        
        self.gamma = 0.99
        self.max_step_per_epi = 200
        self.target_model_update_freq = 20
        self.max_queue_size = 50000
        
        # epsilon 관련 
        self.max_eps = 1
        self.min_eps = 0.1
        self.decay_eps = 400

        self.memory = None
        self.epsilon_vec = []
        self.last_step = 0
        self.last_episode = 0
        self.id = int(time.time())  #time.strftime('%Y%m%d_%H%M%S')

        
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
    def get_epsilon(self, episode):
        eps = self.min_eps+(self.max_eps - self.min_eps)*np.exp(-(episode) / self.decay_eps)
        return eps
    
    def get_action_from_net(self, state):
        with torch.no_grad():
            pred = self.model(torch.tensor([state])) 
            action = pred.max(1)[1]
        return action.item()
    
    def sample_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.get_action_from_net(state)
        return action    
    
    def train_model(self): 
        if self.memory.get_memory_size() < self.batch_size:
            return 
        transition = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transition))
        
        #print(batch) 
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_prime_batch = torch.cat(batch.state_prime)
        done_batch = torch.cat (batch.done)
        
        predictions = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        state_prime_values = self.target_model(state_prime_batch).max(1)[0]
        expected_q_values = (~done_batch* state_prime_values * self.gamma) + reward_batch 
        
        loss = F.mse_loss(predictions, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def put_memory(self, state, action, state_prime, reward, done) : 
        self.memory.push(torch.tensor([state]), 
                         torch.tensor([action], dtype = torch.long), 
                         torch.tensor([state_prime]), 
                         torch.tensor([reward]), 
                         torch.tensor([done], dtype= torch.bool))
    
    def learn(self) : 
        n_actions = self.env.action_space.n
        self.model = DQN(n_actions)
        self.target_model = DQN(n_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)        
        self.memory = ReplayBuffer(self.max_queue_size)
        
        reward_in_epi = 0 
        eps  = 1
        
        for i_epi in range(self.num_episode):
            state = self.env.reset()
            state = state[0]
            if i_epi >= self.warm_episode:
                eps = self.get_epsilon(i_epi - self.warm_episode)
            step = 0 
            while True:
                action = self.sample_action(state, eps)
                state_prime, reward, terminated, truncated, info = self.env.step(action)
                done = (terminated or truncated)
                self.put_memory(state, action, state_prime, reward, done)
                    
                if i_epi >= self.warm_episode:
                    self.train_model()
                    done = (step == self.max_step_per_epi - 1) or done
                else:
                    done = (step == 5 * self.max_step_per_epi - 1) or done
                state = state_prime
                reward_in_epi += reward
                    
                if done:
                    self.epsilon_vec.append(eps)
                    print("episode : " , i_epi, " reward : ", reward_in_epi, " steps : ",  (step+1))
                    self.writer.add_scalar("reward",reward_in_epi , i_epi)
                    self.writer.add_scalar("steps", (step+1), i_epi)

                    reward_in_epi = 0
                    break
                step+=1
                
            if i_epi % self.target_model_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if i_epi % self.save_freq == 0:
                torch.save(self.target_model.state_dict(), f"{self.model_path}/taxi_dqn_{self.id}.pt")
                    
            self.last_episode = i_epi

        
    def predict_action(self,  state) :
        with torch.no_grad():
            pred = self.test_model(torch.tensor([state])) 
            action = pred.max(1)[1]
        return action.item()
 
    def play(self, log_flag = False, sleep=0.2, max_steps=100, model_path = None):
        print(model_path)
        n_actions = self.env.action_space.n
        if model_path is not None:
            self.test_model = DQN(n_actions)
            #self.test_model.load_state_dict(torch.load(model_path)['model_state_dict'])
            self.test_model.load_state_dict(torch.load(model_path))
            self.test_model .eval()
        elif self.target_model is not None :
            self.test_model = DQN(n_actions)
            self.test_model.load_state_dict(self.target_model.state_dict())
            self.test_model .eval()
        else : 
            return 

        
        action_arr = ["down", "up", "right", "left", "pick up", "drop off"]

        i = 0
        state = self.env.reset()    
        state = state[0]
        self.env.render()
        if log_flag:
            print("step: {} - action: ... ".format(i ))
        time.sleep(sleep)
        done = False

        while not done:
            action = self.predict_action(state)
            i += 1
            state, reward, terminated, truncated, info = self.env.step(action)
            done = (terminated or truncated)
            #display.clear_output(wait=True)
            self.env.render()
            if log_flag:
                print("step : {} - action: {}({})".format(i,action_arr[action], action ))
            time.sleep(sleep)
               
            if i == max_steps:
                print("mission : fail")
                break

#env = gym.make("Taxi-v3").env
#agent = Agent(env=env)
#agent.learn()
#env.close()

         
