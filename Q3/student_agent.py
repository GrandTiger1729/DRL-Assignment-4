import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, device):
        super().__init__()
        self.device = device
        self.model = Network(state_dim, action_dim * 2)  # Outputs mean and log_std
        # register action bounds as buffers so .to() moves them
        self.register_buffer(
            'action_low', torch.FloatTensor(action_low)
        )
        self.register_buffer(
            'action_high', torch.FloatTensor(action_high)
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.model(state).chunk(2, dim=-1)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # reparameterization trick
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        # scale to [low, high]
        action = 0.5 * (tanh_action + 1) * (self.action_high - self.action_low) + self.action_low

        # log_prob under original distribution (pre-tanh)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        # correction for tanh's change of variables
        log_prob -= torch.sum(torch.log(1 - tanh_action.pow(2) + 1e-6), dim=-1, keepdim=True)
        return action, log_prob

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """SAC Agent."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        
        self.state_dim = 67  # MuJoCo state space is of shape (67,)
        self.action_dim = 21  # MuJoCo action space is of shape (21,)
        self.action_low = np.array([-1.0] * self.action_dim, dtype=np.float32)
        self.action_high = np.array([1.0] * self.action_dim, dtype=np.float32)
        # Load model
        self.actor = Actor(self.state_dim, self.action_dim, self.action_low, self.action_high, torch.device('cpu'))
        self.actor.load_state_dict(torch.load("actor.pth", map_location=torch.device('cpu')))
        self.actor.eval()
        
    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        action, _ = self.actor.forward(state)
        return action.squeeze(0).detach().numpy()
