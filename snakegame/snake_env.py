import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from snake_game import SnakeGame


class RewardCalculator:
    """Calculate rewards for Snake game"""
    
    def __init__(
        self,
        food_reward: float,
        death_penalty: float,
        step_reward: float,
        distance_reward: float
    ):
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.step_reward = step_reward
        self.distance_reward = distance_reward
    
    def calculate_reward(
        self,
        done: bool,
        self_hit: bool,
        had_food: bool,
        pos_change: float
    ):
        reward = 0.0
        
        # Step reward for alive steps
        if not done: 
            reward += self.step_reward
        
        # Reward for eating food
        if had_food:
            reward += self.food_reward
        
        # Reward for moving closer to food
        if not done and not had_food:
            reward += self.distance_reward * pos_change

        # Penalty for dying
        if done:
            if self_hit:
                # Double penalty for hitting self
                reward += 5 * self.death_penalty
            else:
                # Normal penalty for hitting wall
                reward += self.death_penalty    
        
        return reward


class SnakeEnv(gym.Env):
    """Gymnasium environment wrapper for Snake game with reward calculation"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self, 
        render_mode=None, 
        width=20, 
        height=15,
        food_reward: float = 500.0,
        death_penalty: float = -100.0,
        step_reward: float = 1.0,
        distance_reward: float = 2.0
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.render_mode = render_mode

        # Initialize the game
        self.game = SnakeGame(width=width, height=height)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            food_reward=food_reward,
            death_penalty=death_penalty,
            step_reward=step_reward,
            distance_reward=distance_reward
        )

        # Define action and observation space
        # Actions: 3 relative actions (straight, right, left)
        self.action_space = spaces.Discrete(3)

        # Observation space: 11 boolean/binary features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float32
        )

        # Initialize pygame if render mode is human
        if self.render_mode == "human":
            pygame.init()

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        observation = self.game.reset()
        info = {"score": self.game.score}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        # Take action in game
        observation, terminated, self_hit, had_food, pos_change, truncated, info = self.game.take_action(action)

        # Calculate reward using reward calculator
        reward = self.reward_calculator.calculate_reward(
            done=terminated or truncated,
            self_hit=self_hit,
            had_food=had_food,
            pos_change=pos_change
        )
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Handle pygame events to prevent window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            self.game.render(mode="human")

    def close(self):
        """Close the environment"""
        self.game.close()
