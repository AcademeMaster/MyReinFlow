import numpy as np
import torch
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import os
from collections import deque

# Import original TD3 and improved TD3
import sys
sys.path.append('TD3-master')
from TD3 import TD3 as OriginalTD3
from TD3_Improved import TD3_Improved

def eval_policy(policy, env_name, seed, eval_episodes=10):
    """Evaluate policy performance"""
    eval_env = gym.make(env_name)
    
    # Handle different gym versions for seeding
    try:
        eval_env.seed(seed + 100)
    except AttributeError:
        pass  # New gym versions don't have seed method
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        try:
            state = eval_env.reset(seed=seed + 100)
            if isinstance(state, tuple):
                state = state[0]  # Extract observation from tuple
        except TypeError:
            state = eval_env.reset()
            if isinstance(state, tuple):
                state = state[0]
        
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            step_result = eval_env.step(action)
            
            # Handle different gym versions (4 vs 5 return values)
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                state, reward, done, _ = step_result
            
            avg_reward += reward

    avg_reward /= eval_episodes
    eval_env.close()
    return avg_reward

def plot_training_curves(original_rewards, improved_rewards, original_losses, improved_losses, save_path="training_comparison.png"):
    """Plot training curves comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Evaluation rewards
    ax1.plot(original_rewards, label='Original TD3', alpha=0.8)
    ax1.plot(improved_rewards, label='Improved TD3', alpha=0.8)
    ax1.set_title('Evaluation Rewards')
    ax1.set_xlabel('Evaluation Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Actor losses
    if original_losses['actor'] and improved_losses['actor']:
        ax2.plot(original_losses['actor'], label='Original TD3', alpha=0.7)
        ax2.plot(improved_losses['actor'], label='Improved TD3', alpha=0.7)
        ax2.set_title('Actor Loss')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Actor Loss')
        ax2.legend()
        ax2.grid(True)
    
    # Critic losses
    ax3.plot(original_losses['critic'], label='Original TD3', alpha=0.7)
    ax3.plot(improved_losses['critic'], label='Improved TD3', alpha=0.7)
    ax3.set_title('Critic Loss')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Critic Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Q-values
    ax4.plot(original_losses['q_values'], label='Original TD3 Q1', alpha=0.7)
    ax4.plot(improved_losses['q_values'], label='Improved TD3 Q1', alpha=0.7)
    ax4.set_title('Q-Values')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Q-Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")

def run_experiment(env_name, seed, max_timesteps, start_timesteps, action_chunk_size, policy_type="original"):
    """Run a single experiment"""
    print(f"\nRunning {policy_type} TD3 experiment...")
    
    # Create environment
    env = gym.make(env_name)
    
    # Handle different gym versions for seeding
    try:
        env.seed(seed)
    except AttributeError:
        pass  # New gym versions don't have seed method
    
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize policy
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2 * max_action,
        "noise_clip": 0.5 * max_action,
        "policy_freq": 2,
        "action_chunk_size": action_chunk_size
    }
    
    if policy_type == "original":
        policy = OriginalTD3(**kwargs)
        # Import ReplayBuffer from TD3-master directory
        sys.path.append('TD3-master')
        from utils import ReplayBuffer
        replay_buffer = ReplayBuffer(state_dim, action_dim, action_chunk_size=action_chunk_size)
    else:
        # Add improved TD3 specific parameters
        kwargs.update({
            "offline_critic_update_freq": 5,
            "ema_alpha": 0.01,
            "curriculum_warmup_steps": 10000
        })
        policy = TD3_Improved(**kwargs)
        # Import ReplayBuffer from TD3-master directory
        sys.path.append('TD3-master')
        from utils import ReplayBuffer
        replay_buffer = ReplayBuffer(state_dim, action_dim, action_chunk_size=action_chunk_size)
    
    # Training metrics
    evaluations = []
    actor_losses = []
    critic_losses = []
    q_values = []
    
    # Reset environment with proper gym version handling
    try:
        state = env.reset(seed=seed)
        if isinstance(state, tuple):
            state = state[0]  # Extract observation from tuple
    except TypeError:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
    
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    # Action chunk collection buffers
    action_chunk_buffer = []
    reward_buffer = []
    start_state = None
    
    for t in range(int(max_timesteps)):
        episode_timesteps += 1
        
        # Select action
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * 0.1, size=action_dim)
            ).clip(-max_action, max_action)
        
        # Record start state for action chunk
        if len(action_chunk_buffer) == 0:
            start_state = state.copy()
        
        # Perform action
        step_result = env.step(action)
        
        # Handle different gym versions
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_result
        
        # Calculate done_bool
        try:
            max_steps = env._max_episode_steps
        except AttributeError:
            max_steps = 1000  # Default fallback
        
        done_bool = float(done) if episode_timesteps < max_steps else 0
        
        # Collect action and reward
        action_chunk_buffer.append(action)
        reward_buffer.append(reward)
        
        # Store to replay buffer when chunk is complete or episode ends
        if len(action_chunk_buffer) == action_chunk_size or done:
            # Pad if necessary
            while len(action_chunk_buffer) < action_chunk_size:
                action_chunk_buffer.append(np.zeros(action_dim))
                reward_buffer.append(0.0)
            
            action_chunk = np.array(action_chunk_buffer)
            multi_step_reward = np.array(reward_buffer)
            
            replay_buffer.add(start_state, action_chunk, next_state, multi_step_reward, done_bool)
            
            # Clear buffers
            action_chunk_buffer = []
            reward_buffer = []
        
        state = next_state
        episode_reward += reward
        
        # Train agent
        if t >= start_timesteps:
            loss_dict = policy.train(replay_buffer, batch_size=256)
            if loss_dict:
                critic_losses.append(loss_dict['critic_loss'])
                if loss_dict['actor_loss'] is not None:
                    actor_losses.append(loss_dict['actor_loss'])
                q_values.append(loss_dict['q1_value'])
        
        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            
            # Reset environment
            try:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
            except:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
            
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
            # Clear buffers
            action_chunk_buffer = []
            reward_buffer = []
            
            # Reset action chunk state
            if hasattr(policy, 'reset_action_chunk'):
                policy.reset_action_chunk()
        
        # Evaluate policy
        if (t + 1) % 5000 == 0:
            eval_reward = eval_policy(policy, env_name, seed)
            evaluations.append(eval_reward)
            print(f"Evaluation at step {t+1}: {eval_reward:.3f}")
    
    env.close()
    
    return {
        'evaluations': evaluations,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'q_values': q_values
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=50000, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--action_chunk_size", default=4, type=int)
    parser.add_argument("--compare", action="store_true", help="Compare original and improved TD3")
    parser.add_argument("--improved_only", action="store_true", help="Run only improved TD3")
    args = parser.parse_args()
    
    if args.compare:
        print("Running comparison between Original TD3 and Improved TD3...")
        
        # Run original TD3
        original_results = run_experiment(
            args.env, args.seed, args.max_timesteps, args.start_timesteps, 
            args.action_chunk_size, "original"
        )
        
        # Run improved TD3
        improved_results = run_experiment(
            args.env, args.seed + 1, args.max_timesteps, args.start_timesteps, 
            args.action_chunk_size, "improved"
        )
        
        # Plot comparison
        original_losses = {
            'actor': original_results['actor_losses'],
            'critic': original_results['critic_losses'],
            'q_values': original_results['q_values']
        }
        
        improved_losses = {
            'actor': improved_results['actor_losses'],
            'critic': improved_results['critic_losses'],
            'q_values': improved_results['q_values']
        }
        
        plot_training_curves(
            original_results['evaluations'],
            improved_results['evaluations'],
            original_losses,
            improved_losses
        )
        
        # Print summary
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(f"Original TD3 - Final eval reward: {original_results['evaluations'][-1]:.3f}")
        print(f"Improved TD3 - Final eval reward: {improved_results['evaluations'][-1]:.3f}")
        
        if original_results['actor_losses'] and improved_results['actor_losses']:
            orig_actor_std = np.std(original_results['actor_losses'][-1000:])  # Last 1000 steps
            impr_actor_std = np.std(improved_results['actor_losses'][-1000:])
            print(f"Original TD3 - Actor loss std (last 1000 steps): {orig_actor_std:.6f}")
            print(f"Improved TD3 - Actor loss std (last 1000 steps): {impr_actor_std:.6f}")
            print(f"Actor loss stability improvement: {((orig_actor_std - impr_actor_std) / orig_actor_std * 100):.2f}%")
    
    elif args.improved_only:
        print("Running Improved TD3 only...")
        improved_results = run_experiment(
            args.env, args.seed, args.max_timesteps, args.start_timesteps, 
            args.action_chunk_size, "improved"
        )
        print(f"\nImproved TD3 - Final eval reward: {improved_results['evaluations'][-1]:.3f}")
    
    else:
        print("Running Original TD3 only...")
        original_results = run_experiment(
            args.env, args.seed, args.max_timesteps, args.start_timesteps, 
            args.action_chunk_size, "original"
        )
        print(f"\nOriginal TD3 - Final eval reward: {original_results['evaluations'][-1]:.3f}")

if __name__ == "__main__":
    main()