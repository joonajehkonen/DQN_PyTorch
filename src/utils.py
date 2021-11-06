"""
Here we have utilities for image preprocessing for the CartPole task.


Working directly with raw Atari 2600 frames,which are 210 x 160 pixel images with a 128-colour palette, can be demanding in terms of computation
and memory requirements. We apply a basic preprocessing step aimed at reducing the input dimensionality and dealing with some artefacts of the Atari 2600 emulator. 
First,to encode a single frame we take the maximum value for each pixel colour value over the frame being encoded and the previous frame. 
This was necessary to remove flickering that is present in games where some objects appear only in even frames while other objects appear only in odd frames, 
an artefact caused by the limited number of sprites Atari 2600 can display at once. Second, we then extract the Y channel, also known as luminance, 
from the RGB frame and rescale it to 84 x 84.The function phi from algorithm 1 described below applies this preprocessing to the m most recent frames 
and stacks them to produce the input to the Q-function, in which m = 4, although the algorithm is robust to different values of
m (for example, 3 or 5).

NOTE: In our case: in atari we use the dedicated wrapper for image prepreprocessing.

We can make life simpler and use opencv to resize and grayscale the image (luminance basically means grayscale).

"""
import cv2
import gym
from collections import deque
import numpy as np


def preprocess_frame(img, width, height, inter = cv2.INTER_AREA):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (width, height), interpolation=inter)
    
    return img


# Nicked from: https://github.com/dgriff777/rl_a3c_pytorch/blob/master/environment.py
# And extended it to have the preprocessing step
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=3)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            obs = self.env.render(mode='rgb_array') ## Gets RGB array of shape (x,y,3)
            obs = preprocess_frame(obs, 84, 84) # returns img size (height, width)   
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        obs = self.env.render(mode='rgb_array') ## Gets RGB array of shape (x,y,3)
        obs = preprocess_frame(obs, 84, 84) # returns img size (height, width)   
        self._obs_buffer.append(obs)
        return obs