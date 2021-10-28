'''
    Train and test
'''
#CartPole-v1 https://gym.openai.com/envs/CartPole-v1/

import gym
import numpy as np
from gym import wrappers
from replay_buffer import ReplayBuffer
#from DQN_agent import DQN
import utils
import pyvirtualdisplay



#np.set_printoptions(threshold=np.inf) # 1000 is default


EPISODES = 10
REPLAY_BUFFER_SIZE = 1000000
TIME_STEPS = 10
BATCH_SIZE = 32
GAMMA = 0.99
K_SKIP = 4
LR = 0.00025
INIT_EPSILON = 1
FINAL_EPSILON = 0.05
WIDTH = 84
HEIGHT = 84

pyvirtualdisplay.Display(visible=0, size=(600, 400)).start() # We need this to emulate display on headless server





env = gym.make('CartPole-v1')
n_actions = env.action_space.n
# For saving the video no render
#env = wrappers.Monitor(env, video_callable=False ,force=True)
for i_episode in range(1):
    
    # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1)
    next_state = env.reset() 
    state = env.render(mode='rgb_array') ## Gets RGB array of shape (x,y,3)
    state = utils.preprocess_frame(state, WIDTH, HEIGHT)


    for t in range(1):
        
        print(state)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


