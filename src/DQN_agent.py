'''
    DQN Agent implementation

    DQN Paper: https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf
'''

import torch
import torch.nn as nn

import numpy as np
import tqdm
import wandb


#device = torch.device("cuda:0")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQN(nn.Module):
    def __init__(self, frame_height, frame_width, output_dim):
        
        super(DQN).__init__()

        # TODO: Neural network
        """
        Model architecture:
        We have three convolutional layers following two fully connected layers and a single output for each valid action.
        Each hidden layer is followed by a rectifier nonlinearity max(0,x) which is ReLU().

        The input to the neural network consists of an 84 x 84 x 4 image produced by the preprocessing map phi. 
        The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity ReLU.
        The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
        This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier. 
        The final hidden layer is fully-connected and consists of 512 rectifier units. 
        The output layer is a fully-connected linear layer with a single output for each valid action. 
        The number of valid actions varied between 4 and 18 on the games we considered.
        """

        self.conv_layer1 = nn.Conv2d((frame_height, frame_width), 32, kernel_size=8, stride=4)
        nn.ReLU()
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        nn.ReLU()
        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        nn.ReLU()
        self.linear_layer1 = nn.Linear(64, 512)
        nn.ReLU()
        self.output_layer = nn.Linear(512, output_dim)
    
    def forward(self):
        pass

    