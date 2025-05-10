import gymnasium
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        
        self.state_dim = 5  # Carpole state space is of shape (5,)
        self.action_dim = 1  # Carpole action space is of shape (1,)
        # Load model
        self.actor = Network(self.state_dim, self.action_dim)
        self.actor.load_state_dict(torch.load("best_actor.pth", map_location=torch.device('cpu')))

    def act(self, observation):
        state = torch.FloatTensor(observation)
        action = self.actor(state).detach().numpy()
        return action
