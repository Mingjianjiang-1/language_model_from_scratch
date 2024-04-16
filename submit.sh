#!/bin/bash

#SBATCH --gres=gpu:0
#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --job-name="jiangm-job"
#SBATCH --mem=100G
#SBATCH --open-mode=append
#SBATCH --time=14-0
#SBATCH --partition=john

# srun python -m cs336_basics.train_bpe
srun python -m cs336_basics.tokenizer_encoding