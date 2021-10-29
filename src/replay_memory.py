'''
    Replay memory for DQN
'''

import numpy as np
import random
from collections import deque


# https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3/40182242
# https://stackoverflow.com/questions/62312844/what-is-a-good-approach-for-a-fast-fifo-queue <-- deque should be efficient enough AND can be random.sampled from

class ReplayMemory:
    def __init__(self, size):
        self.replay_memory = deque([], maxlen=size)
        self.size = size

        # According to the DQN algorithm on page 7
        # the memory is a fifo structure of tuples (phi_t, a_t, r_t, phi_t+1)
        
        '''
            Randomly sample minibatches from replay buffer D
        '''
        def sample(self, batch_size):
            transitions = random.sample(self.replay_memory, batch_size)
            tuples_to_batches(transitions, batch_size)
        
        
        '''
            Save transition tuples to our replay buffer D
        '''
        def save(self, state, action, reward, next_state, done):
            transition = (state, action, reward, next_state, done)
            self.replay_memory.append(transition)


        '''
            Check if deque is full
        '''
        def is_full(self):
            return True if self.replay_memory.count == self.size else False


        def tuples_to_batches(self, transitions, batch_size):
            state_batch = np.zeros((batch_size, *(84,84,3)), dtype=np.float32)
            action_batch = np.zeros(batch_size, dtype=np.int32)
            reward_batch = np.zeros(batch_size, dtype=np.float32)
            next_state_batch = np.zeros((batch_size, *(84,84,3)), dtype=np.float32)
            done_batch = np.zeros(batch_size, dtype=np.bool_)

            for idx, transition in enumerate(transitions):
                state_batch[idx] = transition[0]
                action_batch[idx] = transition[1]
                reward_batch[idx] = transition[2]
                next_state_batch[idx] = transition[3]
                done_batch[idx] = transition[4]


            return state_batch, action_batch, reward_batch, next_state_batch, done_batch