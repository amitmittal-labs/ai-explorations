import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        # Two-layer neural network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state: torch.Tensor):
        x = self.fc2(self.relu(self.fc1(state)))
        return self.softmax(x)
    
    def get_action(self, state: torch.Tensor):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        
        # Two-layer neural network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        value = self.fc2(x)
        return value


def calculate_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99
):
    batch_size = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # Calculate returns in reverse order
    running_return = 0
    for t in reversed(range(batch_size)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
        advantages[t] = returns[t] - values[t]
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def calculate_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    # Calculate probability ratio: r(theta) = pi(a|s) / pi_old(a|s)
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Calculate surrogate losses
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    
    # Take the minimum (pessimistic bound)
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
    return policy_loss


def calculate_value_loss(values: torch.Tensor, returns: torch.Tensor):
    value_loss = 0.5 * (values - returns).pow(2).mean()
    return value_loss


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
    ):
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
    def reset_memory(self):
        """Reset memory buffers"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
    def select_action(self, state: np.ndarray):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action(state_tensor)
            value = self.value_net(state_tensor).item()
        
        return action, log_prob.item(), value
    
    def store_transition(self, state, action, log_prob, reward, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
    
    def update(self, epochs: int = 10):
        if len(self.states) == 0:
            return {}
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        old_values = torch.FloatTensor(self.values)
        
        # Calculate advantages and returns
        advantages, returns = calculate_advantages(rewards, old_values, self.gamma)
        
        # PPO update epochs
        for _ in range(epochs):
            # Get current policy log probs
            probs = self.policy_net(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            
            # Get current values
            values = self.value_net(states).squeeze()
            
            # Calculate losses
            policy_loss = calculate_policy_loss(
                log_probs, old_log_probs, advantages, self.clip_epsilon
            )
            value_loss = calculate_value_loss(values, returns)
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Clear memory after update
        self.reset_memory()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save(self, filepath: str, save_full: bool = False):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save the model
            save_full: If True, saves both policy and value networks (for resuming training)
                      If False, saves only policy network (for inference only)
        """
        if save_full:
            # Save everything (for resuming training)
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'value_state_dict': self.value_net.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict()
            }, filepath)
            print(f"Full model saved to {filepath} (can resume training)")
        else:
            # Save only policy network (for inference)
            torch.save({
                'policy_state_dict': self.policy_net.state_dict()
            }, filepath)
            print(f"Policy model saved to {filepath} (inference only)")
    
    def load(self, filepath: str, inference_only: bool = False):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to the saved model
            inference_only: If True, only loads policy network (for playing/evaluation)
                           If False, loads everything (for resuming training)
        """
        checkpoint = torch.load(filepath)
        
        # Always load policy network
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        
        if not inference_only:
            # Load value network and optimizers (for training)
            if 'value_state_dict' in checkpoint:
                self.value_net.load_state_dict(checkpoint['value_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
                print(f"Full model loaded from {filepath} (ready for training)")
            else:
                print(f"Warning: Only policy network found in {filepath}")
                print(f"Value network not loaded - cannot resume training")
        else:
            print(f"Policy model loaded from {filepath} (inference mode)")
