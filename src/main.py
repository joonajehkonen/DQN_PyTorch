'''
    Train and test
'''
#CartPole-v1 https://gym.openai.com/envs/CartPole-v1/

import gym
from gym import wrappers
from DQN_agent import DQNAgent
import utils
import pyvirtualdisplay



EPISODES = 25000#10000#100000
TIME_STEPS = 300

pyvirtualdisplay.Display(visible=0, size=(600, 400)).start() # We need this to emulate display on headless server

# For saving the video no render
#env = wrappers.Monitor(env, video_callable=False ,force=True)

env = gym.make('CartPole-v1')
env = utils.MaxAndSkipEnv(env) # On cartpole we can use this extended wrapper for image preprocessing
env = wrappers.FrameStack(env, 4)
#env = gym.make('Pong-v4')
# NOTE: OpenAI gym has a atari_preprocessing wrapper which implements the whole image preprocessing
"""
env = wrappers.AtariPreprocessing(
    env,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=False,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=True #False
)
"""
# Get current env action space size (number of actions)
n_actions = env.action_space.n
TRAIN = True
policy_name = '/media/data1/joojeh/DQN_PyTorch/policies/dqn_{}.pth'.format(EPISODES)

dqn_agent = DQNAgent(EPISODES, TIME_STEPS, n_actions, (4, 84, 84), policy_name)

if TRAIN:
    dqn_agent.train(env)
else:
    dqn_agent.evaluate()

env.close()
