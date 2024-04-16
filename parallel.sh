#!/bin/bash

bs=(0.005 0.001 0.0001 0.00003)
# bs=(64 128 256 512)

for seed in "${bs[@]}"; do
	sbatch lm_training.sh $seed
done

 