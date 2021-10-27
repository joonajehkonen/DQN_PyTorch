'''
    Train and test
'''
#CartPole-v1 https://gym.openai.com/envs/CartPole-v1/

import gym

from replay_buffer import ReplayBuffer






EPISODES = 10
REPLAY_BUFFER_SIZE = 1000000
T = 10
BATCH_SIZE = 32

# Initialize replay memory D to capacity N
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Initialize action-value function Q with random weights h

# Initialize target action-value function Q^ with weights h2 5 h

for _ in EPISODES:
    pass

    # Initialize sequence s_1 = {x1} and preprocessed sequence phi_1 = phi(s_1)

    for _ in T:
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
