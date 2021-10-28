"""
Here we have utilities for image preprocessing.


Working directly with raw Atari 2600 frames,which are 210 x 160 pixel images with a 128-colour palette, can be demanding in terms of computation
and memory requirements. We apply a basic preprocessing step aimed at reducing the input dimensionality and dealing with some artefacts of the Atari 2600 emulator. 
First,to encode a single frame we take the maximum value for each pixel colour value over the frame being encoded and the previous frame. 
This was necessary to remove flickering that is present in games where some objects appear only in even frames while other objects appear only in odd frames, 
an artefact caused by the limited number of sprites Atari 2600 can display at once. Second, we then extract the Y channel, also known as luminance, 
from the RGB frame and rescale it to 84 x 84.The function phi from algorithm 1 described below applies this preprocessing to the m most recent frames 
and stacks them to produce the input to the Q-function, in which m = 4, although the algorithm is robust to different values of
m (for example, 3 or 5).


We can make life simpler and use opencv to resize and grayscale the image (luminance basically means grayscale).

On cartpole we use preprocessing without the encoding part because there is no flickering.

"""

import cv2


def preprocess_encoded_frame(img, width, height, inter = cv2.INTER_AREA):
    img = encode_frame()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height), interpolation=inter)

    return img

def preprocess_frame(img, width, height, inter = cv2.INTER_AREA):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (width, height), interpolation=inter)

    img /= 255.0 # normalize to range [0,1]

    return img



def encode_frame():
    pass