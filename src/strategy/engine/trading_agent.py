# Import libaries

from collections import deque
from random import sample

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

####################################################################################################
# Define the Double Deep Q-learning Agent (DDQNAgent) class
class DDQNAgent:
    """
    A class to implement the Double Deep Q-learning Agent.
    """
    def __init__(self, config: dict, logger, state_dimension: int, action_dimension: int, device: torch.device):
        """
        Initialize the DDQNAgent class.
        """
        # Initialize the class variables
        self.config = config
        self.logger = logger
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        replay_capacity = self.config['strategy']['replay_capacity']
        self.replay_memory = deque([],maxlen=replay_capacity)
        self.learning_rate = self.config['strategy']['learning_rate']
        self.gamma = self.config['strategy']['gamma']
        self.architecture = self.config['strategy']['architecture']
        self.l2_reg = self.config['strategy']['l2_reg']
        self.device = device

        # Define the two neural networks
        self.online_model = self.build_model()
        self.target_model = self.build_model(trainable = False)

        # move the models to the device
        self.online_model.to(self.device)
        self.target_model.to(self.device)

        # Update the target model
        self.update_target_model()

        # Define the loss function and the optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)

        # Define the \epsilon-greedy parameters
        self.epsilon = self.config['strategy']['epsilon_start']
        self.epsilon_decay_steps = self.config['strategy']['epsilon_decay_steps']
        self.epsilon_decay = (self.config['strategy']['epsilon_start'] - self.config['strategy']['epsilon_end']) / self.epsilon_decay_steps
        self.epsilon_exponential_decay = self.config['strategy']['epsilon_exponential_decay']
        self.epsilon_history = []

        # Initialize the training parameters
        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        # Define the model parameters
        self.batch_size = self.config['strategy']['batch_size']
        self.tau = self.config['strategy']['tau']
        self.losses = []
        self.idx = torch.arange(self.batch_size)
        self.train = True
        

    def build_model(self, trainable = True):
        """
        Build the neural network model.
        """
        model = nn.Sequential()
        for i, layer in enumerate(self.architecture):
            # Add the input layer
            if i == 0:
                model.add_module(f'layer_{i}', nn.Linear(self.state_dimension, layer))
            else:
                model.add_module(f'layer_{i}', nn.Linear(self.architecture[i-1], layer))
            # Add the activation function
            if i < len(self.architecture) - 1:
                model.add_module(f'relu_{i}', nn.ReLU())
        # Add the dropout layer
        model.add_module('dropout', nn.Dropout(p=self.config['strategy']['dropout']))
        # Add the output layer
        model.add_module('output', nn.Linear(self.architecture[-1], self.action_dimension))
        # Optionally freeze parameters if not trainable
        if not trainable:
            for param in model.parameters():
                param.requires_grad = False

        return model
        
        

    def update_target_model(self):
        """"
        Update the target model with the online model.
        """
        self.target_model.load_state_dict(self.online_model.state_dict())
        

    def epsilon_greedy(self, state):
        """"
        Implement the epsilon-greedy policy
        """
        self.total_steps += 1
        if self.epsilon > np.random.rand():
            action = np.random.randint(self.action_dimension)
        else:
            q = self.online_model(torch.tensor(state, dtype=torch.float32)).detach()
            action = torch.argmax(q).item()
        return action

    def memorize_transition(self, state, action, reward, next_state, not_done):
        """"
        Store the transition in the replay memory.
        """
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0

        self.replay_memory.append((state, action, reward, next_state, not_done))

    def experience_replay(self):
        if len(self.replay_memory) < self.batch_size:
            return
        mini_batch = map(np.array, zip(*sample(self.replay_memory, self.batch_size)))
        # maps to the device
        states, actions, rewards, next_states, not_done = mini_batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # Add extra dimension
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        not_done = torch.tensor(not_done, dtype=torch.float32, device=self.device).unsqueeze(1)


        # Compute the Q-values and the target Q-values
        current_q_values = self.online_model(states)
        with torch.no_grad():
            best_actions = self.online_model(next_states).argmax(dim=1, keepdim=True)  # Online model selects actions
            max_next_q_values = self.target_model(next_states).gather(1, best_actions)  # Target model estimates Q-value
            target_q_values = rewards + not_done * self.gamma * max_next_q_values

        # Get Q-values for chosen actions
        q_values = current_q_values.gather(1, actions)

        # Compute loss
        loss = self.criterion(q_values, target_q_values)
        self.losses.append(loss.item())
        
        # Optimize the online model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Update the target model after tau steps
        if self.total_steps % self.tau == 0:
            self.update_target_model()



        