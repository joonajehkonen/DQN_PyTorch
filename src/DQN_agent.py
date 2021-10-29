'''
    DQN Agent implementation

    DQN Paper: https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf
'''

import torch
import torch.nn as nn

import numpy as np
import tqdm
import wandb
import random

from replay_memory import ReplayMemory
import utils

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
        ).to(device)


    def forward(self, x):
        x = x.to(device)
        return self.dqn(x)


    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            #m.weight.data.normal_(0.0, 0.02)
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(1.0, 0.02)
            #m.bias.data.fill_(0)
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        

"""
EPISODES = 10
REPLAY_MEMORY_SIZE = 1000000
TIME_STEPS = 10
BATCH_SIZE = 32
GAMMA = 0.99
K_SKIP = 4
LR = 0.00025
INIT_EPSILON = 1
FINAL_EPSILON = 0.05
WIDTH = 84
HEIGHT = 84
"""

"""
Implementation of the agent (Algorithm 1).
"""
class DQNAgent:
    def __init__(self, episodes, time_steps, n_actions, input_dim, epsilon=1.0, final_epsilon=0.05, memory_size=1000000, 
                 gamma=0.99, lr=0.00025, batch_size=32):
        self.dqn = DQN(input_dim, n_actions) 
        self.target_dqn = DQN(input_dim, n_actions) 
        self.replay_memory = ReplayMemory(memory_size) # Initialize replay memory D to capacity N
        self.action_space = [i for i in range(n_actions)] # possible_actions = [list(range(1, (k + 1))) for k in action_space.nvec] https://stackoverflow.com/questions/64588828/openai-gym-walk-through-all-possible-actions-in-an-action-space
        self.episodes = episodes
        self.time_steps = time_steps
        self.total_reward = 0
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.frame_skip = 4
        self.update_freq = 10000
        self.width = 84
        self.height = 84
        #elf.optimizer = torch.optim.Adam(theta,lr)
        #self.loss = nn.MSELoss(reduction=mean)
        
        pass

    """
    Policy for choosing actions.
    Exploration / eploitation is handled with epsilon greedy and
    policy estimation is done with Deep Q Network.
    """
    def policy(self, state, actions):
        p = np.random.random()

        if p < self.epsilon:
            action = random.choice(actions)
        else:
            # TODO: state into torch tensor
            action = torch.argmax(self.dqn.forward(state))
            #action = torch.argmax(actions)
        
        return action


    # TODO: Frame skipping every 4th frame
    def train(self, env):
        # Initialize action-value function Q with random weights h
        self.dqn.apply(self.dqn.init_weights)
        # Initialize target action-value function Q^ with weights h⁻ = h
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        for i_episode in range(self.episodes):

            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1)
            env.reset() 
            state = env.render(mode='rgb_array') ## Gets RGB array of shape (x,y,3)
            state = utils.preprocess_frame(state, self.width, self.height)


            for t in range(self.time_steps):
                # With probability E select a random action a_t
                # otherwise select a_t = argmax_Q(phi(s_t), a; theta)
                action = self.policy(state, self.action_space)

                # Execute action a_t in emulator and observe reward r_t and image x_t+1
                next_state, reward, done, _ = env.step(action)

                # TODO: is necessary? Reward between -1 and 1
                reward = max(-1.0, min(reward, 1.0))

                self.total_reward += reward

                # Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state = utils.preprocess_frame(next_state, self.width, self.height)
                
                # Store transition (phi_t, a_t, r_t, phi_t+1) in D
                if self.replay_memory.is_full():
                    self.replay_memory.popleft() # Remove last item on queue
                    self.replay_memory.save(state, action, reward, next_state, done)
                
                state = next_state



                """
                Training of DQN
                """

                # TODO: Go here when enough samples in memory
                # TODO: consider the architecture of replay buffer

                # Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D
                next_state_batch, action_batch, reward_batch, state_batch, done_batch = self.replay_memory.sample(self.batch_size)
                
                next_state_batch = torch.from_numpy(next_state_batch) # NumPy array to torch tensor for NN
                state_batch = torch.from_numpy(state_batch) # NumPy array to torch tensor for NN
                
                # Set y_j = r_j                                             if episode terminates at step j + 1
                #     y_j = r_j + gamma * max_a' Q_hat(phi_j+1,a'; theta⁻)  otherwise

                if done:
                    #total_reward = reward_batch
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                else:
                    next_state, reward, done, _ = self.target_dqn(next_state_batch).detach()
                    #forward(next_state_batch).detach()
                    target = reward_batch + reward * self.gamma

                # Perform a gradient descent step on (y_j - Q(phi_j,a_j; theta))² with respect to the network parameters theta
                next_state, reward, done, _ = self.dqn(state_batch)
                #forward(state_batch)
                loss = (target - reward) ** 2
                #self.optimizer.zero_grad()

                #loss.backward()

                #self.optimizer.step()
                #self.optimizer.zero_grad()

                # Every C steps reset Q_theta = Q
                if t % self.update_freq == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())
            
                
    


    

    def evaluate():
        pass
    
    


"""

# Initialize replay memory D to capacity N
replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# Initialize action-value function Q with random weights h

# Initialize target action-value function Q^ with weights h⁻ = h

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