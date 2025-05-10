# Setup logging to train.log
import logging

import random
import numpy as np
import gymnasium as gym
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

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
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = Network(state_dim + action_dim, 1)  # Q-value network

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, transition):
        state, action, reward, next_state, done = transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device).unsqueeze(1),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device).unsqueeze(1)
        )

    def __len__(self):
        return self.size

class SAC:
    def __init__(self, state_dim, action_dim, action_low, action_high, *,
                 capacity=100000,
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # initialize actor and critics
        self.actor = Actor(state_dim, action_dim, action_low, action_high, self.device).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)

        # copy weights to targets
        for targ, src in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            targ.data.copy_(src.data)
        for targ, src in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            targ.data.copy_(src.data)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # replay buffer
        self.buffer = ReplayBuffer(capacity, state_dim, action_dim)
        self.best_score = -np.inf

    def save_checkpoint(self, filename, episode, steps):
        ckpt = {
            'episode': episode,
            'steps': steps,
            'best_score': self.best_score,
            'actor_state': self.actor.state_dict(),
            'critic1_state': self.critic1.state_dict(),
            'critic2_state': self.critic2.state_dict(),
            'target_critic1_state': self.target_critic1.state_dict(),
            'target_critic2_state': self.target_critic2.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic1_opt': self.critic1_optimizer.state_dict(),
            'critic2_opt': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha.data.clone(),
            'alpha_opt': self.alpha_optimizer.state_dict()
        }
        torch.save(ckpt, filename)
        logging.info(f"Saved checkpoint {filename} @ ep {episode}, step {steps}")

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename, weights_only=False, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state'])
        self.critic1.load_state_dict(ckpt['critic1_state'])
        self.critic2.load_state_dict(ckpt['critic2_state'])
        self.target_critic1.load_state_dict(ckpt['target_critic1_state'])
        self.target_critic2.load_state_dict(ckpt['target_critic2_state'])
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic1_optimizer.load_state_dict(ckpt['critic1_opt'])
        self.critic2_optimizer.load_state_dict(ckpt['critic2_opt'])
        self.log_alpha.data.copy_(ckpt.get('log_alpha', torch.zeros_like(self.log_alpha)))
        self.alpha_optimizer.load_state_dict(ckpt.get('alpha_opt', self.alpha_optimizer.state_dict()))
        self.alpha = self.log_alpha.exp()
        self.best_score = ckpt.get('best_score', self.best_score)
        logging.info(f"Loaded checkpoint {filename}")

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action, _ = self.actor(state)
                # Scale action to [low, high]
                action = 0.5 * (action + 1) * (self.actor.action_high - self.actor.action_low) + self.actor.action_low
            else:
                action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        # target Q
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            tq1 = self.target_critic1(next_state, next_action)
            tq2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(tq1, tq2) - self.alpha * next_log_prob
            backup = reward + (1 - done) * self.gamma * target_q

        # current Q
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        loss_q1 = F.mse_loss(current_q1, backup)
        loss_q2 = F.mse_loss(current_q2, backup)

        self.critic1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # actor
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # soft updates
        for targ, src in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            targ.data.copy_(self.tau * src.data + (1 - self.tau) * targ.data)
        for targ, src in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            targ.data.copy_(self.tau * src.data + (1 - self.tau) * targ.data)

    def train(self, env: gym.Env, *, num_episodes=10000, update_every=8, validate_every=50, checkpoint_dir="checkpoints"): 
        os.makedirs(checkpoint_dir, exist_ok=True)
        steps = 0
        for ep in tqdm(range(num_episodes)):
            state, _ = env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                self.buffer.push((state, action, reward, next_state, done))
                state = next_state
                steps += 1

                if steps % update_every == 0:
                    self.update()

            if ep % validate_every == 0:
                mean_r, std_r = self.test(env, num_episodes=20)
                logging.info(f"Episode {ep}, Steps {steps}, Mean reward: {mean_r:.3f} ± {std_r:.3f}")
                current_performance = mean_r - std_r
                if current_performance > self.best_score:
                    self.best_score = current_performance
                    best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
                    self.save_checkpoint(best_ckpt_path, ep, steps)
                    # Save the actor model separately
                    torch.save(self.actor.state_dict(), "actor.pth")
                    logging.info(f"New best performance: mean - std = {current_performance:.3f}")
                current_ckpt_path = os.path.join(checkpoint_dir, "current_checkpoint.pth")
                self.save_checkpoint(current_ckpt_path, ep, steps)

    def test(self, env: gym.Env, num_episodes=20) -> tuple[np.float32, np.float32]:
        self.actor.eval()
        rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                action = self.select_action(state, deterministic=True)
                next_state, reward, term, trunc, _ = env.step(action)
                total += reward
                done = term or trunc
                state = next_state
            rewards.append(total)
        self.actor.train()
        return np.mean(rewards), np.std(rewards)
    
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def make_env(env_name: str, seed: int, flatten: bool = True, use_pixels: bool = False) -> gym.Env:
    set_seed(seed)
    env = make_dmc_env(env_name, seed, flatten=flatten, use_pixels=use_pixels)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent with checkpoint support.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    args = parser.parse_args()

    env_name = "humanoid-walk"
    seed = 89487
    
    env = make_env(env_name, seed, flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    logging.info(f"Environment: {env_name}, State dim: {state_dim}, Action dim: {action_dim}, Action low: {action_low}, Action high: {action_high}")
    
    sac_agent = SAC(state_dim, action_dim, action_low, action_high, capacity=int(1e6), batch_size=512)
    if args.resume:
        sac_agent.load_checkpoint(args.resume)
    # sac_agent.train(env, num_episodes=50000, update_every=1, validate_every=50)
    torch.save(sac_agent.actor.state_dict(), "actor.pth")
