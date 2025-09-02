import numpy as np
import torch
import gym
import argparse
import os
import minari

import utils
import TD3_BC


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	# Load Minari dataset and recover environment
	minari_dataset = minari.load_dataset(env_name)
	eval_env = minari_dataset.recover_environment(eval_env=True)
	# Set seed for evaluation environment
	eval_env.reset(seed=seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset()
		done = False
		while not done:
			# Handle different state formats
			if isinstance(state, tuple):
				# For newer gym versions that return tuple
				state_obs = state[0] if isinstance(state, tuple) and len(state) > 0 else state
			else:
				# For direct state observation
				state_obs = state
			
			# Ensure state is a numpy array with correct shape
			state_array = np.asarray(state_obs, dtype=np.float32)
			if state_array.ndim == 0:
				state_array = state_array.reshape(1, -1)
			elif state_array.ndim == 1:
				state_array = state_array.reshape(1, -1)
			state_normalized = (state_array - mean) / std
			action = policy.select_action(state_normalized)
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done = terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="mujoco/pusher/expert-v0") # Minari environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	# Replace slashes in filename to avoid path issues
	file_name = file_name.replace("/", "_")
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("models"):
		os.makedirs("models")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	# Load Minari dataset and recover environment
	minari_dataset = minari.load_dataset(args.env)
	env = minari_dataset.recover_environment()
	
	# Set seeds
	env.reset(seed=args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	policy = TD3_BC.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_minari(minari_dataset)
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	# 保存归一化参数以便后续使用
	if args.normalize:
		np.savez(f"models/{file_name}_norm", mean=mean, std=std)
	
	evaluations = []
	for t in range(int(args.max_timesteps)):
		policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
			np.save(f"models/{file_name}", evaluations)
			if args.save_model: 
				policy.save(f"./models/{file_name}")
