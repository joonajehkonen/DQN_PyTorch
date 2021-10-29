'''
    Train and test
'''
#CartPole-v1 https://gym.openai.com/envs/CartPole-v1/

import gym
import numpy as np
from gym import wrappers
from DQN_agent import DQNAgent
from replay_memory import ReplayMemory
#from DQN_agent import DQN
import utils
import pyvirtualdisplay



#np.set_printoptions(threshold=np.inf) # 1000 is default


EPISODES = 10
#REPLAY_MEMORY_SIZE = 1000000
TIME_STEPS = 10
#BATCH_SIZE = 32
#GAMMA = 0.99
#K_SKIP = 4
#LR = 0.00025
#INIT_EPSILON = 1
#FINAL_EPSILON = 0.05
WIDTH = 84
HEIGHT = 84


pyvirtualdisplay.Display(visible=0, size=(600, 400)).start() # We need this to emulate display on headless server





env = gym.make('CartPole-v1')
n_actions = env.action_space.n
TRAIN = True
policy_name = 'policies/dqn_{EPISODES}'
# For saving the video no render
#env = wrappers.Monitor(env, video_callable=False ,force=True)

"""
episodes, time_steps, n_actions, input_dim, output_dim, epsilon=1.0, final_epsilon=0.05, buffer_memory=1000000, 
                 gamma=0.99, lr=0.00025, batch_size=32):
"""
dqn_agent = DQNAgent(EPISODES, TIME_STEPS, n_actions, (WIDTH, HEIGHT, 3), n_actions)

if TRAIN:
    dqn_agent.train()
else:
    dqn_agent.evaluate()

env.close()


