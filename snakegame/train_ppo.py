"""
Training script for Snake game using custom PPO implementation
Rewards are calculated by the environment (snake_env.py)
"""

import numpy as np
import torch
from snake_env import SnakeEnv
from model import PPOAgent
import time
from collections import deque


def train_ppo(
    total_timesteps: int = 500000,
    n_steps: int = 2048,
    epochs: int = 10,
    render: bool = False,
    log_freq: int = 10,
    eval_freq: int = 10000,
    eval_episodes: int = 5,
    # Environment reward parameters
    food_reward: float = 10.0,
    death_penalty: float = -10.0,
    step_penalty: float = -0.01
):
    print("=" * 60)
    print("Training Snake with Custom PPO Implementation")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Steps per update: {n_steps}")
    print(f"PPO epochs: {epochs}")
    print("-" * 60)
    print("Environment Reward Parameters:")
    print(f"  Food reward: {food_reward}")
    print(f"  Death penalty: {death_penalty}")
    print(f"  Step penalty: {step_penalty}")
    print("-" * 60)
    
    # Create environment with reward parameters
    render_mode = "human" if render else None
    env = SnakeEnv(
        render_mode=render_mode,
        food_reward=food_reward,
        death_penalty=death_penalty,
        step_penalty=step_penalty
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print("-" * 60)
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr_policy=3e-4,
        lr_value=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        value_coef=0.5
    )
    
    # Training statistics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_count = 0
    timestep = 0
    
    # Reset environment
    state, info = env.reset()
    episode_reward = 0
    episode_length = 0
    
    start_time = time.time()
    last_eval_timestep = 0
    
    print("Starting training...")
    print("-" * 60)
    
    while timestep < total_timesteps:
        # Collect trajectories
        for step in range(n_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment (environment calculates reward)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition (using reward from environment)
            agent.store_transition(state, action, log_prob, reward, value)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            timestep += 1
            
            # Move to next state
            state = next_state
            
            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # Log episode statistics
                if episode_count % log_freq == 0:
                    avg_reward = np.mean(episode_rewards)
                    avg_length = np.mean(episode_lengths)
                    elapsed_time = time.time() - start_time
                    fps = timestep / elapsed_time
                    
                    print(f"Episode {episode_count} | Timestep {timestep}/{total_timesteps}")
                    print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                    print(f"  Avg Length (last 100): {avg_length:.1f}")
                    print(f"  Last Score: {info.get('score', 0)}")
                    print(f"  FPS: {fps:.1f}")
                    print("-" * 60)
                
                # Reset environment
                state, info = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Break if we've reached total timesteps
            if timestep >= total_timesteps:
                break
        
        # Update PPO agent
        update_stats = agent.update(epochs=epochs)
        
        if update_stats:
            print(f"Update at timestep {timestep}")
            print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {update_stats['value_loss']:.4f}")
            print("-" * 60)
        
        # Evaluate model
        if timestep - last_eval_timestep >= eval_freq:
            eval_scores = evaluate_agent(agent, env, eval_episodes, render=False)
            avg_eval_score = np.mean(eval_scores)
            print(f"Evaluation at timestep {timestep}")
            print(f"  Average Score: {avg_eval_score:.2f}")
            print(f"  Scores: {eval_scores}")
            print("-" * 60)
            last_eval_timestep = timestep
    
    # Save inference-only model (smaller, just policy network)
    agent.save("snake_ppo_model_final.pt")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    final_scores = evaluate_agent(agent, env, 20, render=False)
    print(f"Final Evaluation (20 episodes):")
    print(f"  Average Score: {np.mean(final_scores):.2f}")
    print(f"  Best Score: {max(final_scores)}")
    print(f"  Scores: {final_scores}")
    print()
    print("Model saved:")
    print(f"  snake_ppo_model_final.pt   - Policy only (for inference)")
    
    env.close()
    
    return agent


def evaluate_agent(agent: PPOAgent, env: SnakeEnv, num_episodes: int = 5, render: bool = False):
    scores = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action (only uses policy network)
            action, _, _ = agent.select_action(state)
            
            # Take step (environment calculates reward)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                env.render()
        
        scores.append(info.get('score', 0))
    
    return scores


def play_trained_model(model_path: str = "snake_ppo_model_final.pt", episodes: int = 5):
    print("=" * 60)
    print(f"Loading model: {model_path}")
    print("=" * 60)
    
    # Create environment with rendering (default rewards)
    env = SnakeEnv(render_mode="human")
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load model (inference only - just policy network)
    agent.load(model_path)
    
    print(f"Watching trained agent play {episodes} episodes...")
    print("Close the window to stop early")
    print("-" * 60)
    
    scores = []
    
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        print(f"Episode {episode + 1}/{episodes}")
        
        while not done:
            # Get action from model (only uses policy network)
            action, _, _ = agent.select_action(state)
            
            # Take step (environment calculates reward)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Render
            env.render()
            time.sleep(0.05)  # Slow down for viewing
        
        score = info.get('score', 0)
        scores.append(score)
        print(f"  Score: {score}, Reward: {episode_reward:.2f}")
        print("-" * 60)
    
    env.close()
    
    print("\nResults:")
    print(f"  Average Score: {np.mean(scores):.2f}")
    print(f"  Best Score: {max(scores)}")
    print(f"  All Scores: {scores}")
    
    return scores


def main():
    """Main function with command line interface"""
    import sys
    
    if len(sys.argv) == 1:
        # Default training
        print("Starting default training...")
        train_ppo()
    
    elif sys.argv[1] == "train":
        # Custom training
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
        render = "--render" in sys.argv
        
        print(f"Training for {timesteps} timesteps...")
        if render:
            print("Rendering enabled")
        
        train_ppo(
            total_timesteps=timesteps,
            render=render
        )
    
    elif sys.argv[1] == "play":
        # Play trained model (inference only)
        model_path = sys.argv[2] if len(sys.argv) > 2 else "snake_ppo_model_final.pt"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        play_trained_model(model_path, episodes)
    
    elif sys.argv[1] == "eval":
        # Evaluate trained model (inference only)
        model_path = sys.argv[2] if len(sys.argv) > 2 else "snake_ppo_model_final.pt"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        
        print(f"Evaluating model: {model_path}")
        
        env = SnakeEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(model_path, inference_only=True)  # Only load policy network
        
        scores = evaluate_agent(agent, env, episodes, render=False)
        
        print("\nEvaluation Results:")
        print(f"  Episodes: {episodes}")
        print(f"  Average Score: {np.mean(scores):.2f}")
        print(f"  Std Score: {np.std(scores):.2f}")
        print(f"  Min Score: {min(scores)}")
        print(f"  Max Score: {max(scores)}")
        print(f"  All Scores: {scores}")
        
        env.close()
    
    else:
        print("Usage:")
        print("  python train_ppo.py                          # Train with defaults")
        print("  python train_ppo.py train 200000             # Train for 200k steps")
        print("  python train_ppo.py train 50000 --render     # Train with rendering")
        print("  python train_ppo.py play                     # Watch trained model")
        print("  python train_ppo.py play model.pt 3          # Watch specific model (3 games)")
        print("  python train_ppo.py eval                     # Evaluate trained model")
        print("  python train_ppo.py eval model.pt 20         # Evaluate specific model (20 games)")


if __name__ == "__main__":
    main()
