#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --job-name="jiangm-job"
#SBATCH --mem=100G
#SBATCH --open-mode=append
#SBATCH --time=14-0
#SBATCH --partition=sphinx

srun python -m cs336_basics.train --lr $1 --group_name 'ts_lr_search'
# srun python -m cs336_basics.train --lr 0.001 --batch_size $1 --group_name 'ts_bs_search'
