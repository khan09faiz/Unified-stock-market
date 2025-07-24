"""
Reinforcement Learning Module for Stock Trading.
This module contains the PPO agent and trading environment for stock market analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class PolicyNetwork(nn.Module):
    """Policy network for PPO agent."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) + 0.15)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.fc(state)
        std = torch.exp(self.log_std.clamp(min=-2, max=0.6))
        return mean, std


class ValueNetwork(nn.Module):
    """Value network for PPO agent."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc(state)


class StockTradingEnv(gym.Env):
    """
    Stock trading environment for reinforcement learning.
    Supports continuous action space for position sizing.
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000,
                 stop_loss_pct: float = 0.02, episode_length: int = 300,
                 trend_reward_coef: float = 0.02, sentiment_reward_coef: float = 0.01):
        super(StockTradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.episode_length = episode_length
        self.trend_reward_coef = trend_reward_coef
        self.sentiment_reward_coef = sentiment_reward_coef
        
        # Observation space: price changes + volatility + sentiment + trend
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Action space: continuous position from -1 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Trading state
        self.reset()
        
        # Performance tracking
        self.stop_loss_count = 0
        self.reward_history = []
        self.peak_net_worth = initial_capital
        self.trade_history = []

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = np.random.randint(0, max(1, len(self.data) - self.episode_length))
        self.start_step = self.current_step
        self.net_worth = self.initial_capital
        self.position = 0.0  # Current position (-1 to 1)
        self.prev_action = 0.0
        self.prev_total_assets = self.initial_capital
        self.reward_history = []
        self.peak_net_worth = self.initial_capital
        self.trade_history = []
        
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        if self.current_step >= len(self.data):
            return np.zeros(8, dtype=np.float32)
        
        # Get recent price changes (last 5 steps)
        start = max(0, self.current_step - 4)
        recent_data = self.data.iloc[start:self.current_step + 1]
        
        if 'Average_Price_Change' in recent_data.columns:
            price_changes = recent_data['Average_Price_Change'].values
        else:
            # Fallback: calculate price changes from Close prices
            if 'End_Price' in recent_data.columns:
                prices = recent_data['End_Price'].values
            elif 'Close' in recent_data.columns:
                prices = recent_data['Close'].values
            else:
                prices = np.ones(len(recent_data)) * 100  # Default price
            
            price_changes = np.diff(prices, prepend=prices[0]) / prices[0] if len(prices) > 0 else np.array([0])
        
        # Pad if necessary
        if len(price_changes) < 5:
            price_changes = np.pad(price_changes, (5 - len(price_changes), 0), mode='constant')
        
        # Scale price changes
        scaled_prices = price_changes / 0.15
        
        # Get current market state
        row = self.data.iloc[self.current_step]
        
        # Volatility mapping - ensure we get scalar values
        volatility_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': 1}
        try:
            volatility_level = row['Volatility_Level'] if 'Volatility_Level' in row.index else 'Medium'
            if isinstance(volatility_level, pd.Series):
                volatility_level = volatility_level.iloc[0] if len(volatility_level) > 0 else 'Medium'
            volatility = volatility_map.get(str(volatility_level), 1) / 2.0
        except:
            volatility = 0.5  # Default
        
        # Sentiment score (normalized)
        try:
            sentiment_score = row['Sentiment_Score'] if 'Sentiment_Score' in row.index else 0
            if isinstance(sentiment_score, pd.Series):
                sentiment_score = sentiment_score.iloc[0] if len(sentiment_score) > 0 else 0
            sentiment = np.clip(float(sentiment_score), -2, 2) / 2
        except:
            sentiment = 0.0  # Default
        
        # Trend mapping
        trend_map = {'Downtrend': -1, 'Stable': 0, 'Uptrend': 1}
        try:
            trend_classification = row['Trend_Classification'] if 'Trend_Classification' in row.index else 'Stable'
            if isinstance(trend_classification, pd.Series):
                trend_classification = trend_classification.iloc[0] if len(trend_classification) > 0 else 'Stable'
            trend = trend_map.get(str(trend_classification), 0)
        except:
            trend = 0  # Default
        
        return np.concatenate([
            scaled_prices,
            [volatility, sentiment, trend]
        ], dtype=np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step."""
        action = float(np.clip(action, -1, 1))
        
        if self.current_step >= len(self.data):
            return self._get_observation(), 0, True, {}
        
        row = self.data.iloc[self.current_step]
        
        # Get current price
        try:
            if 'End_Price' in row.index:
                current_price = float(row['End_Price'])
            elif 'Close' in row.index:
                current_price = float(row['Close'])
            else:
                current_price = 100.0  # Default price
        except:
            current_price = 100.0  # Default price
        
        # Get trend and sentiment with proper handling
        trend_map = {'Downtrend': -1, 'Stable': 0, 'Uptrend': 1}
        try:
            trend_class = row['Trend_Classification'] if 'Trend_Classification' in row.index else 'Stable'
            if isinstance(trend_class, pd.Series):
                trend_class = trend_class.iloc[0] if len(trend_class) > 0 else 'Stable'
            trend = trend_map.get(str(trend_class), 0)
        except:
            trend = 0
            
        try:
            sentiment = row['Sentiment_Score'] if 'Sentiment_Score' in row.index else 0
            if isinstance(sentiment, pd.Series):
                sentiment = sentiment.iloc[0] if len(sentiment) > 0 else 0
            sentiment = float(sentiment)
        except:
            sentiment = 0.0
        
        # Execute trade
        delta_position = action - self.position
        transaction_cost = abs(delta_position) * current_price * 0.002  # 0.2% transaction cost
        
        if delta_position > 0:  # Buying
            buy_cost = delta_position * current_price + transaction_cost
            if self.net_worth >= buy_cost:
                self.net_worth -= buy_cost
                self.position += delta_position
        elif delta_position < 0:  # Selling
            sell_gain = -delta_position * current_price - transaction_cost
            self.net_worth += sell_gain
            self.position += delta_position
        
        # Ensure position bounds
        self.position = np.clip(self.position, 0, 1)
        
        # Calculate total assets
        unrealized_value = self.position * current_price
        total_assets = self.net_worth + unrealized_value
        self.peak_net_worth = max(self.peak_net_worth, total_assets)
        
        # Calculate reward
        reward = self._calculate_reward(action, delta_position, total_assets, trend, sentiment, row)
        
        # Record trade
        self.trade_history.append({
            'step': self.current_step,
            'action': action,
            'position': self.position,
            'price': current_price,
            'net_worth': self.net_worth,
            'total_assets': total_assets,
            'reward': reward
        })
        
        # Check termination conditions
        done = False
        
        # Stop loss
        if total_assets <= self.initial_capital * (1 - self.stop_loss_pct):
            done = True
            self.stop_loss_count += 1
            reward -= 1.0  # Penalty for stop loss
        
        # Episode length or data end
        self.current_step += 1
        if (self.current_step >= len(self.data) - 1 or 
            self.current_step - self.start_step >= self.episode_length):
            done = True
        
        self.prev_total_assets = total_assets
        self.prev_action = action
        
        next_state = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        info = {
            'net_worth': self.net_worth,
            'position': self.position,
            'total_assets': total_assets,
            'current_price': current_price
        }
        
        return next_state, reward, done, info

    def _calculate_reward(self, action: float, delta_position: float, total_assets: float,
                         trend: float, sentiment: float, row: pd.Series) -> float:
        """Calculate reward for the current step."""
        # Base reward: change in portfolio value
        self.reward_history.append((total_assets - self.prev_total_assets) / self.initial_capital)
        
        # Smoothed profit over last 10 steps
        if len(self.reward_history) > 10:
            smoothed_profit = np.mean(self.reward_history[-10:])
        else:
            smoothed_profit = np.mean(self.reward_history)
        
        reward = smoothed_profit
        
        # Trend alignment reward
        reward += self.trend_reward_coef * (action * trend)
        
        # Sentiment alignment reward
        reward += self.sentiment_reward_coef * (action * sentiment)
        
        # Transaction cost penalty
        reward -= 0.0005 * abs(delta_position)
        
        # Position size penalty (encourage moderate positions)
        reward -= 0.00025 * action**2
        
        # Volatility penalty
        volatility_map = {'Low': 0, 'Medium': 0.0005, 'High': 0.001, 'Unknown': 0.0005}
        try:
            vol_level = row['Volatility_Level'] if 'Volatility_Level' in row.index else 'Medium'
            if isinstance(vol_level, pd.Series):
                vol_level = vol_level.iloc[0] if len(vol_level) > 0 else 'Medium'
            volatility_penalty = volatility_map.get(str(vol_level), 0.0005)
        except:
            volatility_penalty = 0.0005
        reward -= volatility_penalty
        
        # Profit protection reward
        if delta_position < 0 and total_assets < self.prev_total_assets:
            reward += 0.002 * abs(delta_position)  # Reward cutting losses
        
        # Peak protection reward
        if total_assets > self.peak_net_worth * 0.95 and delta_position < 0:
            reward += 0.003 * abs(delta_position)  # Reward taking profits near peak
        
        return reward

    def render(self, mode: str = 'human'):
        """Render the current state."""
        if self.current_step < len(self.data):
            print(f"Step: {self.current_step}, "
                  f"Net Worth: {self.net_worth:.2f}, "
                  f"Position: {self.position:.2f}, "
                  f"Total Assets: {self.net_worth + self.position * self.data.iloc[self.current_step].get('End_Price', 100):.2f}")

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of trading performance."""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(self.trade_history)
        
        total_return = (df['total_assets'].iloc[-1] - self.initial_capital) / self.initial_capital
        max_drawdown = (df['total_assets'].cummax() - df['total_assets']).max() / self.initial_capital
        
        # Sharpe ratio calculation
        returns = df['total_assets'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len([t for t in self.trade_history if abs(t['action'] - self.prev_action) > 0.1]),
            'final_position': self.position,
            'stop_loss_triggered': self.stop_loss_count > 0
        }


class PPOAgent:
    """Proximal Policy Optimization agent for stock trading."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 lr_value: float = 1e-3, gamma: float = 0.99, eps_clip: float = 0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=100, gamma=0.97)

    def select_action(self, state: np.ndarray) -> Tuple[float, torch.Tensor]:
        """Select action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy_old(state)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action.clamp(-3, 3))
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return action.squeeze().detach().numpy(), log_prob.squeeze()

    def update(self, memory: Dict[str, List]) -> Dict[str, float]:
        """Update policy and value networks."""
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.FloatTensor(np.array(memory['actions']))
        old_log_probs = torch.stack(memory['log_probs']).detach()
        rewards = memory['rewards']
        
        # Calculate returns
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns)
        
        # Calculate advantages
        values = self.value(states).squeeze().detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        policy_losses = []
        value_losses = []
        
        for _ in range(10):  # Multiple epochs
            mean, std = self.policy(states)
            mean = torch.clamp(mean, -0.375, 0.375)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            values = self.value(states).squeeze()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(values, returns)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss = policy_loss + 0.5 * value_loss - 0.2 * entropy
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        # Update exploration noise
        self.policy.log_std.data = torch.clamp(self.policy.log_std.data - 0.011, min=-2, max=0.6)
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.lr_scheduler.step()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': entropy.item()
        }

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, filepath)

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
