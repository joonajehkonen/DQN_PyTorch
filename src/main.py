'''
    Train and test
'''
#CartPole-v1 https://gym.openai.com/envs/CartPole-v1/

import gym
import utils
from gym import wrappers
from DQN_agent import DQNAgent
import pyvirtualdisplay


#PongNoFrameskip
#CartPole-v1
ENV_NAME = 'CartPole-v1'
pyvirtualdisplay.Display(visible=0, size=(210, 160)).start() # We need this to emulate display on headless server
env = gym.make(ENV_NAME)

if ENV_NAME == 'CartPole-v1':
    EPISODES = 25000
    # NOTE: # On cartpole we can use these extended wrappers for help in image preprocessing
    
    env = utils.MaxAndSkipEnv(env) #
    env = wrappers.FrameStack(env, 4)
else:
    EPISODES = 12500000
    # NOTE: OpenAI gym has a atari_preprocessing wrapper which implements the whole image preprocessing
    # NOTE: I use these wrappers to ensure the preprocessing is done correctly
    # NOTE: We need to use the NoFrameSkip version because we set frame skipping in our env wrapper!
    env = wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False # NOTE: Let's do scaling after sampling to save memory (this way we can save obs as uint8)
    )
    env = wrappers.FrameStack(env, 4)

# For saving the video no render
#env = wrappers.Monitor(env, video_callable=False ,force=True)

# Get current env action space size (number of actions)
n_actions = env.action_space.n
TRAIN = True
POLICY_NAME = '/media/data1/joojeh/DQN_PyTorch/policies/dqn_{}_{}_ep.pth'.format(ENV_NAME, EPISODES)
MODEL_PATH = '/media/data1/joojeh/DQN_PyTorch/policies/dqn_cartpole_adam_25000ep.pth'
RECORD_PATH = '/media/data1/joojeh/DQN_PyTorch/recordings'

dqn_agent = DQNAgent(EPISODES, n_actions, (4, 84, 84), POLICY_NAME, ENV_NAME)

if TRAIN:
    dqn_agent.train(env)
else:
    env = wrappers.RecordVideo(env, video_folder=RECORD_PATH , name_prefix='dqn_cartpole_adam_25000ep')
    dqn_agent.evaluate(env, 5,  MODEL_PATH)

env.close()
