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

"""
Implementation of the DeepQNetwork (DQN)
"""
class DQN(nn.Module):
    def __init__(self, in_channels, output_dim):
        
        super().__init__()

        input_dim = in_channels[0]

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

        I use here PyTorch sequential module, which simplifies the forward function quite a bit.
        """

        dqn = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = x.to(device)
        return self.dqn(x)
        

"""
Implementation of the agent (Algorithm 1).
"""
class Agent:
    def __init__(self):
        pass


    def policy():
        pass


    def train():
        pass

    
    def eval():
        pass
    
    


"""

# Initialize replay memory D to capacity N
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Initialize action-value function Q with random weights h

# Initialize target action-value function Q^ with weights h2 5 h

for _ in EPISODES:
    pass

    # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1)

    for _ in TIME_STEPS:
        pass
        # With probability E select a random action a_t
        # otherwise select a_t = argmax_Q(phi(s_t), a; theta)
        
        # Execute action a_t in emulator and observe reward r_t and image x_t+1
        
        # Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)

        # Store transition (phi_t, a_t, r_t, phi_t+1) in D

        # Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D

        # Set y_j = r_j                                             if episode terminates at step j + 1
        #     y_j = r_j + gamma * max_a' Q_hat(phi_j+1,a'; theta⁻)  otherwise

        # Perform a gradient descent step on (y_j - Q(phi_j,a_j; theta))² with respect to the network parameters theta

        # Every C steps reset Q_theta = Q
"""