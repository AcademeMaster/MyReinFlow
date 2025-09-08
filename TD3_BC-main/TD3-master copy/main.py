import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import wandb

import utils
import TD3



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.reset(seed=seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(state))
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
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Pendulum-v1")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=10, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--action_chunk_size", default=4, type=int)  # Action chunk size for action chunking
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--use_wandb", action="store_true",default=True)         # Use wandb for logging
	parser.add_argument("--wandb_project", default="TD3-Training")   # Wandb project name
	parser.add_argument("--wandb_entity", default="ucas-feng")       # Wandb entity name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	# Initialize wandb
	if args.use_wandb:
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			name=f"{file_name}_{wandb.util.generate_id()}",
			config={
				"policy": args.policy,
				"env": args.env,
				"seed": args.seed,
				"start_timesteps": args.start_timesteps,
				"eval_freq": args.eval_freq,
				"max_timesteps": args.max_timesteps,
				"expl_noise": args.expl_noise,
				"batch_size": args.batch_size,
				"discount": args.discount,
				"tau": args.tau,
				"policy_noise": args.policy_noise,
				"noise_clip": args.noise_clip,
				"policy_freq": args.policy_freq,
				"action_chunk_size": args.action_chunk_size
			}
		)

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["action_chunk_size"] = args.action_chunk_size
		policy = TD3.TD3(**kwargs)


	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, action_chunk_size=args.action_chunk_size)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, _ = env.reset()
	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# 用于收集action chunks的临时存储
	action_chunk_buffer = []
	reward_buffer = []
	start_state = None

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# 如果是新的action chunk的开始，记录起始状态
		if len(action_chunk_buffer) == 0:
			start_state = state.copy()

		# Perform action
		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# 收集action和reward
		action_chunk_buffer.append(action)
		reward_buffer.append(reward)

		# 当收集到足够的action chunk或episode结束时，存储到replay buffer
		if len(action_chunk_buffer) == args.action_chunk_size or done:
			# 填充不足的部分（如果episode提前结束）
			while len(action_chunk_buffer) < args.action_chunk_size:
				action_chunk_buffer.append(np.zeros(action_dim))  # 用零填充
				reward_buffer.append(0.0)  # 用零填充reward
			
			# 转换为numpy数组
			action_chunk = np.array(action_chunk_buffer)
			multi_step_reward = np.array(reward_buffer)
			
			# 存储到replay buffer
			replay_buffer.add(start_state, action_chunk, next_state, multi_step_reward, done_bool)
			
			# 清空缓冲区
			action_chunk_buffer = []
			reward_buffer = []

		state = next_state
		episode_reward += reward

		# Log step reward to wandb
		if args.use_wandb:
			wandb.log({
				"timestep": t,
				"step_reward": reward,
				"cumulative_episode_reward": episode_reward
			})

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			loss_dict = policy.train(replay_buffer, args.batch_size)
			# Log training metrics to wandb
			if args.use_wandb and loss_dict:
				wandb.log({
					"timestep": t,
					"critic_loss": loss_dict['critic_loss'],
					"actor_loss": loss_dict['actor_loss'],
					"q1_value": loss_dict['q1_value'],
					"q2_value": loss_dict['q2_value'],
					"target_q": loss_dict['target_q']
				})

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			
			# Log episode metrics to wandb
			if args.use_wandb:
				wandb.log({
					"timestep": t+1,
					"episode_num": episode_num+1,
					"episode_reward": episode_reward,
					"episode_length": episode_timesteps
				})
			
			# Reset environment
			state, _ = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			# 清空缓冲区
			action_chunk_buffer = []
			reward_buffer = []
			# 重置动作序列状态
			if hasattr(policy, 'reset_action_chunk'):
				policy.reset_action_chunk() 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			eval_reward = eval_policy(policy, args.env, args.seed)
			evaluations.append(eval_reward)
			np.save(f"./results/{file_name}", evaluations)
			
			# Log evaluation metrics to wandb
			if args.use_wandb:
				wandb.log({
					"timestep": t+1,
					"eval_reward": eval_reward,
					"eval_episode": len(evaluations)
				})
			
			if args.save_model: policy.save(f"./models/{file_name}")

	# Visualize trained policy
	print("\n" + "="*50)
	print("Training completed! Starting visualization...")
	print("="*50)
	
	# Create environment with rendering
	render_env = gym.make(args.env, render_mode="human")
	render_env.reset(seed=args.seed + 200)
	
	# Run visualization episodes
	num_vis_episodes = 5
	for episode in range(num_vis_episodes):
		print(f"\nVisualization Episode {episode + 1}/{num_vis_episodes}")
		state, _ = render_env.reset()
		done = False
		episode_reward = 0
		step_count = 0
		
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = render_env.step(action)
			done = terminated or truncated
			episode_reward += reward
			step_count += 1
			
		print(f"Episode {episode + 1} - Reward: {episode_reward:.3f}, Steps: {step_count}")
	
	render_env.close()
	print("\nVisualization completed!")
	
	# Finish wandb run
	if args.use_wandb:
		wandb.finish()
