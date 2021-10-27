'''
    Replay buffer for DQN
'''

import numpy as np
import random



class ReplayBuffer:
    def __init__(self, size):
        self.size = size

        # TODO: Replay buffer
        replay_buffer = np.zeros(self.size)

        # TODO: We need way to init, store and sample according to the DQN algorithm on page 7
        # TODO: The memory is a list of tuples (phi_t, a_t, r_t, phi_t+1)
        
        '''
            Randomly sample minibatches from replay buffer D
        '''
        def sample(self, batch_size):
            return random.sample(self.replay_buffer, batch_size)
        
        
        '''
            Save transition tuples to our replay buffer D
        '''
        def store(self, state, action, reward, next_state):
            transition = (state, action, reward, next_state)
            self.replay_buffer.append(transition)
