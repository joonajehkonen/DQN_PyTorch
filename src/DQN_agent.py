'''
    DQN Agent implementation

    DQN Paper: https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf
'''

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
import wandb
import random
import matplotlib.pyplot as plt

from replay_memory import ReplayMemory

"""
Implementation of the Deep Q Network (DQN)
"""
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        """
        Model architecture:
        We have three convolutional layers following two fully connected layers and a single output for each valid action.
        Each hidden layer is followed by a rectifier nonlinearity max(0,x) which is ReLU().

        The input to the neural network consists of an 4 x 84 x 84 image produced by the preprocessing map phi. 
        The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity ReLU.
        The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
        This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier. 
        The final hidden layer is fully-connected and consists of 512 rectifier units. 
        The output layer is a fully-connected linear layer with a single output for each valid action. 
        The number of valid actions varied between 4 and 18 on the games we considered.

        I use here PyTorch sequential module, which simplifies the architecture implementation quite a bit.
        """

        n_input_channels = input_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(512, num_actions)

    def forward(self, observations: torch.Tensor) -> (torch.Tensor):
        x = self.cnn(observations)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.final_fc(x)
        return x



"""
Implementation of the agent (Algorithm 1).
"""
class DQNAgent:
    def __init__(self, episodes, n_actions, input_shape, policy_name, env_name, epsilon=1.0, final_epsilon=0.05, memory_size=10000, 
                 gamma=0.99, lr=0.00025, batch_size=32):

        # Initialize action-value function Q with random weights h
        self.dqn = DQN(input_shape, n_actions).type(torch.cuda.FloatTensor)
        self.target_dqn = DQN(input_shape, n_actions).type(torch.cuda.FloatTensor)
        
        # NOTE: This is done on training loop when episode is 0 ==> Initialize target action-value function Q^ with weights h⁻ = h

        # NOTE: RMSprop for Atari Pong and Adam for CartPole
        if (env_name == 'CartPole-v1'):
            self.optimizer = torch.optim.Adam(params=self.dqn.parameters(), lr=lr)
            self.decay_over_frames = 6500
        else:
            self.optimizer = torch.optim.RMSprop(params=self.dqn.parameters(), lr=lr, alpha=0.95, weight_decay=0, eps=0.01,  momentum=0, centered=True)
            self.decay_over_frames = 20000

        # Paper uses MSE as loss
        # NOTE: in MSE the order of substraction doesn't matter. Paper uses (target - prediction)². Pytorch uses (prediction - target)²
        self.loss_func = nn.MSELoss()

        # Initialize replay memory D to capacity N
        self.replay_memory = ReplayMemory(memory_size)
        self.memory_size = memory_size
        self.policy_name = policy_name
        self.action_space = [i for i in range(n_actions)]
        self.episodes = episodes
        self.total_reward = 0
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.update_freq = 1000
        self.episodic_rewards = []
        self.mean_episodic_rewards = []
        self.losses = []
        self.episodic_losses = []
        self.start_training = 1000


    """
    Policy for choosing actions.
    Exploration / exploitation is handled with epsilon greedy and
    policy estimation is done with Deep Q Network.
    """
    def policy(self, state):
        if np.random.rand() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            state = torch.from_numpy(np.asarray(state) / 255.0).unsqueeze(0).type(torch.cuda.FloatTensor) # LazyFrame to torch tensor
            action = torch.argmax(self.dqn(state)).cpu().item()
        
        return action


    def train(self, env):
        
        # Wandb config for logging
        config = {
            "Learning rate": self.lr,
            "Episodes": self.episodes,
            "Batch size": self.batch_size,
            "Update freq.": self.update_freq,
            "Replay memory size": self.memory_size,
            "Gamma": self.gamma
        }

        run = wandb.init(
            project="PyTorch_DQN", entity="joonaj",
            config = config,
            monitor_gym=True,
            save_code=True
        )
        
        iter_no = 0
    
        for i_episode in tqdm(range(self.episodes)):
            self.total_reward = 0
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1)
            done = False
            state = env.reset()

            while not done:
                
                # With probability E select a random action a_t
                # otherwise select a_t = argmax_Q(phi(s_t), a; theta)
                action = self.policy(state)

                # Execute action a_t in emulator and observe reward r_t and image x_t+1
                # Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state, reward, done, _ = env.step(action)
                
                # Clip reward between [-1, 1], not using torch clamp for np arrays
                self.total_reward += reward
                reward = np.clip(reward, -1.0, 1.0)

                # Store transition (phi_t, a_t, r_t, phi_t+1) in D
                if self.replay_memory.is_full():
                    self.replay_memory.popleft() # Remove last item on queue
                    self.replay_memory.save(state, action, reward, next_state, done)
                else:
                    self.replay_memory.save(state, action, reward, next_state, done)
                
                # Save current state for next iteration
                state = next_state
                iter_no += 1

                """
                Training of DQN when enough samples in memory
                """
                if iter_no >= self.start_training:

                    # Epsilon decay
                    if self.epsilon > 0.05:
                        self.epsilon = self.epsilon - (0.95 / self.decay_over_frames)

                    # Sample random minibatch of transitions (phi_j, a_j, r_j, phi_j+1) from D
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.sample(self.batch_size)
                    
                    next_state_batch = torch.from_numpy(next_state_batch / 255.0).type(torch.cuda.FloatTensor)  # NumPy array to torch tensor for NN
                    state_batch = torch.from_numpy(state_batch / 255.0).type(torch.cuda.FloatTensor) # NumPy array to torch tensor for NN
                    reward_batch = torch.from_numpy(reward_batch).cuda() # NumPy array to torch tensor for NN and for gpu
                    action_batch = torch.from_numpy(action_batch).long().cuda() # NumPy array to torch tensor for NN and for gpu.
                    done_batch = torch.from_numpy(done_batch).cuda() # NumPy array to torch tensor for NN and for gpu

                    # Get current state Q values (predictions) (Tensor shape of [32] ==> This is achieved by getting 2D inout to gather and squeezing it to 1D after)
                    current_q = self.dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    # Set y_j = r_j                                             if episode terminates at step j + 1 NOTE: we can check this with (not) done mask
                    #     y_j = r_j + gamma * max_a' Q_hat(phi_j+1,a'; theta⁻)  otherwise
                    # The not done mask works like this: When the j + 1 is done, the done will be 1 ==> 1 - 1 = 0 ==> reward_batch + 0
                    target_max_q = self.target_dqn(next_state_batch).detach().max(1)[0]
                    target = reward_batch + target_max_q * self.gamma * (1 - done_batch) 

                    # Perform a gradient descent step on (y_j - Q(phi_j,a_j; theta))² with respect to the network parameters theta
                    loss = self.loss_func(current_q, target)
                    
                    # Add loss to array of losses. NOTE: loss is a tensor
                    self.losses.append(loss) 

                    # Clip loss between [-1,1], we can use torch.clamp on torch tensors
                    loss = loss.clamp(-1,1)

                    # Standard PyTorch update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                if done:
                    self.episodic_rewards.append(self.total_reward)
                    break

            # Every C steps reset Q_theta = Q
            if i_episode % self.update_freq == 0:
                print("Target network updated!")
                self.target_dqn.load_state_dict(self.dqn.state_dict())
            
            if i_episode % 100 == 0 and iter_no  > self.start_training:
                print("Episode: {}".format(i_episode))
                print("Total Mean Episodic reward: {0:.4f}".format(sum(self.episodic_rewards) / len(self.episodic_rewards)))
                print("Total MSE: {0:.4f}".format(torch.mean(torch.stack(self.losses), dim=0)))
                self.mean_episodic_rewards.append(sum(self.episodic_rewards) / len(self.episodic_rewards))
                self.episodic_losses.append(torch.mean(torch.stack(self.losses), dim=0))
            elif i_episode % 100 == 0 and iter_no <= self.start_training:
                print("Episode: {}".format(i_episode))

        

        # Plot logging to wandb
        episodic_rewards_np = np.array(self.mean_episodic_rewards)
        plt.plot(np.arange(len(episodic_rewards_np)), episodic_rewards_np, ls="solid")
        plt.ylabel("Episodic reward")
        plt.xlabel("Every 100th episode")
        wandb.log({"chart": plt})

        # Turn list of tensors to numpy array with scalars for plotting
        total_loss = []
        for loss in self.episodic_losses:
            total_loss.append(loss.cpu().item())

        total_loss_np = np.array(total_loss)
        plt.plot(np.arange(len(total_loss_np)), total_loss_np, ls="solid")
        plt.ylabel("Loss")
        plt.xlabel("Every 100th episode")
        wandb.log({"chart": plt})
        
        # Save trained model
        torch.save(self.dqn.state_dict(), self.policy_name)
        run.finish()
                
            
    """
    Same structure as in training, but no model training part.
    """
    def evaluate(self, env, run_episodes, model_path, record_path):

        self.epsilon = -1 # No random actions on evaluation
        trained_model = torch.load(model_path)
        self.dqn = torch.load_state_dict(trained_model)

        for i_episode in tqdm(range(run_episodes)):
            self.total_reward = 0
            iter_no = 0
            # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1)
            done = False
            state = env.reset()

            while not done:
                
                # With probability E select a random action a_t
                # otherwise select a_t = argmax_Q(phi(s_t), a; theta)
                action = self.policy(state)

                # Execute action a_t in emulator and observe reward r_t and image x_t+1
                # Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)
                next_state, reward, done, _ = env.step(action)
                
                # Clip reward between [-1, 1], not using torch clamp for np arrays
                self.total_reward += reward
                
                # Save current state for next iteration
                state = next_state
                iter_no += 1
                    
                if done:
                    self.episodic_rewards.append(self.total_reward)
                    print('Episode: {}, steps: {}, reward: {} mean episodic reward: {}'.format(i_episode, iter_no, self.total_reward,
                        self.episodic_rewards / i_episode))
                    break
    