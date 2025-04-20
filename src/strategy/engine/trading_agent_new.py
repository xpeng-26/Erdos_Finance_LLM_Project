# Import libaries

from collections import deque
from random import sample

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

####################################################################################################
# Define the Double Deep Q-learning Agent (DDQNAgent) class
# Add other features
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

        self.is_multi_discrete = isinstance(action_dimension, np.ndarray)
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
        self.idx = torch.arange(self.batch_size, device=self.device)
        self.train = True

    def build_model(self, trainable=True):
        """Build the neural network model."""
        if self.is_multi_discrete:
            return self.build_multi_head_model(trainable)
        else:
            return self.build_single_head_model(trainable)
        

    def build_single_head_model(self, trainable = True):
        """
        Build model for Discrete action space.
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
    

    def build_multi_head_model(self, trainable=True):
        """Build model for MultiDiscrete action space."""
        class MultiHeadNetwork(nn.Module):
            def __init__(self, state_dim, action_dims, architecture, dropout):
                super().__init__()
                # Shared layers
                self.shared = nn.ModuleList()
                current_dim = state_dim[0]*state_dim[1]
                for units in architecture:
                    self.shared.append(nn.Linear(current_dim, units))
                    self.shared.append(nn.ReLU())
                    current_dim = units
                
                self.dropout = nn.Dropout(p=dropout)
                # Separate output head for each action dimension
                self.heads = nn.ModuleList([
                    nn.Linear(architecture[-1], dim) for dim in action_dims
                ])

            def forward(self, x):
                # Flatten the input state tensor (batch_size, 4, 36) -> (batch_size, 144)
                x = x.view(x.size(0), -1)  # Flatten the state tensor
                
                # Forward through shared layers
                for layer in self.shared:
                    x = layer(x)
                x = self.dropout(x)

                # Get output from each head
                outputs = [head(x) for head in self.heads]
                return outputs

        model = MultiHeadNetwork(
            self.state_dimension,
            self.action_dimension,
            self.architecture,
            self.config['strategy']['dropout']
        )

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
        """Implement the epsilon-greedy policy."""
        self.total_steps += 1
        if self.epsilon > np.random.rand():
            if self.is_multi_discrete:
                return [np.random.randint(dim) for dim in self.action_dimension]
            else:
                return np.random.randint(self.action_dimension)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.online_model(state_tensor)
                if self.is_multi_discrete:
                    return [q.argmax().item() for q in q_values]
                else:
                    return q_values.argmax().item()


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
        
        # Sample a mini-batch and convert it to tensors efficiently
        mini_batch = sample(self.replay_memory, self.batch_size)
        # Unpack mini_batch and ensure proper tensor conversion
        states, actions, rewards, next_states, not_done = zip(*mini_batch)

        # Convert states and next_states properly (stacking ensures correct shape)
        states = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in states])
        next_states = torch.stack([torch.tensor(ns, dtype=torch.float32, device=self.device) for ns in next_states])

        # Convert actions, rewards, not_done (ensuring they are tensors with the right shape)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        not_done = torch.tensor(not_done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Update the multi-discrete actions model
        if self.is_multi_discrete:
            actions = [torch.tensor(a, dtype=torch.long, device=self.device) for a in zip(*actions)]
            current_q_values = self.online_model(states)
            current_q_selected = [q.gather(1, a.unsqueeze(1)) for q, a in zip(current_q_values, actions)]
            
            with torch.no_grad():
                next_q_values = self.online_model(next_states)
                best_actions = [q.argmax(dim=1, keepdim=True) for q in next_q_values]
                target_q_values = self.target_model(next_states)
                max_next_q_values = [q.gather(1, a) for q, a in zip(target_q_values, best_actions)]
                target_q = [rewards + not_done * self.gamma * nq for nq in max_next_q_values]

            loss = sum(self.criterion(cq, tq) for cq, tq in zip(current_q_selected, target_q))

        # Update the discrete actions model
        else:
            # The actions
            actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            # Compute Q-values
            current_q_values = self.online_model(states).gather(1, actions)

            with torch.no_grad():
                best_actions = self.online_model(next_states).argmax(dim=1, keepdim=True)
                max_next_q_values = self.target_model(next_states).gather(1, best_actions)
                target_q_values = rewards + not_done * self.gamma * max_next_q_values

            # Compute loss
            loss = self.criterion(current_q_values, target_q_values)
        
        
        # Store the loss
        self.losses.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target model
        if self.total_steps % self.tau == 0:
            self.update_target_model()
