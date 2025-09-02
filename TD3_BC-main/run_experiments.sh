#!/bin/bash

# Script to reproduce models

envs=(
	"mujoco/halfcheetah-random-v0"
	"mujoco/hopper-random-v0"
	"mujoco/walker2d-random-v0"
	"mujoco/halfcheetah-medium-v0"
	"mujoco/hopper-medium-v0"
	"mujoco/walker2d-medium-v0"
	"mujoco/halfcheetah-expert-v0"
	"mujoco/hopper-expert-v0"
	"mujoco/walker2d-expert-v0"
	"mujoco/halfcheetah-medium-expert-v0"
	"mujoco/hopper-medium-expert-v0"
	"mujoco/walker2d-medium-expert-v0"
	"mujoco/halfcheetah-medium-replay-v0"
	"mujoco/hopper-medium-replay-v0"
	"mujoco/walker2d-medium-replay-v0"
	)

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		python main.py \
		--env $env \
		--seed $i
	done
done