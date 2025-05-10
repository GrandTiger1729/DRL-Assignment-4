# Setup logging to train.log
import logging
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import gymnasium as gym
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

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
    
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.actor = Network(state_dim, action_dim)
        self.critic = Network(state_dim, 1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state):
        mean = self.actor(state)
        dist = MultivariateNormal(mean, torch.eye(self.action_dim))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def rollout(self, env: gym.Env, num_episodes=10):
        states, actions, total_rewards, log_probs = [], [], [], []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            rewards = []
            while not done:
                action, log_prob = self.select_action(torch.FloatTensor(state))
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                state = next_state
                done = terminated or truncated
            
            total_reward = 0
            tmp = []
            for t in reversed(range(len(rewards))):
                total_reward = rewards[t] + self.gamma * total_reward
                tmp.append(total_reward)
            tmp.reverse()
            total_rewards.extend(tmp)
            
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        total_rewards = torch.FloatTensor(total_rewards)
        log_probs = torch.stack(log_probs)
            
        return states, actions, total_rewards, log_probs
    
    def train(self, env: gym.Env, num_iterations=1000, num_episodes_per=10, num_updates_per=10, validate_per=50):
        best_mean = -np.inf
        
        for iteration in tqdm(range(num_iterations), desc="Training"):
            states, actions, total_rewards, log_probs = self.rollout(env, num_episodes=num_episodes_per)
            
            advantages = total_rewards - self.critic(states).squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(num_updates_per):
                # Update critic
                values = self.critic(states).squeeze()
                critic_loss = nn.MSELoss()(values, total_rewards)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # Update actor
                new_mean = self.actor(states)
                new_dist = MultivariateNormal(new_mean, torch.eye(self.action_dim))
                new_log_probs = new_dist.log_prob(actions)
                ratio = (new_log_probs - log_probs).exp()
                
                surrogate1 = ratio * (total_rewards - values.detach())
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * (total_rewards - values.detach())
                
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            
            if (iteration + 1) % validate_per == 0:
                logging.info(f"Iteration {iteration}: Training loss: {actor_loss.item():.3f}, Critic loss: {critic_loss.item():.3f}")
                mean = self.validate(env, num_episodes=10)
                # Early stopping condition
                if mean > best_mean:
                    best_mean = mean
                    logging.info(f"Iteration {iteration}: New best mean reward: {mean:.3f}")
                    torch.save(self.actor.state_dict(), "best_actor.pth")
                    torch.save(self.critic.state_dict(), "best_critic.pth")
                
            
    def validate(self, env: gym.Env, num_episodes=10):
        total_rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.actor(torch.FloatTensor(state)).detach().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                state = next_state
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        logging.info(f"Validation: Mean reward: {mean_reward:.3f}, Std reward: {std_reward:.3f}")
        
        return mean_reward
                
if __name__ == "__main__":
    # Create environment
    env = make_dmc_env("cartpole-balance", np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logging.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim)
    ppo_agent.train(env, num_iterations=400, num_episodes_per=10, num_updates_per=10, validate_per=50)
