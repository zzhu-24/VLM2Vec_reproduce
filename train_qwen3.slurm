#!/bin/bash
#SBATCH --job-name=tr            # Name of your job
#SBATCH --output=./slurm_results/%x/%j.out
#SBATCH --error=./slurm_results/%x/%j.err
#SBATCH --partition=RTX3090              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:3                 # Request 2 GPUs
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --ntasks-per-node=3
#SBATCH --mem=30G                     # Request 32 GB of memory
#SBATCH --time=48:00:00               # Time limit for the job (hh:mm:ss)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vlm2vec
cd ~/2025/VLM2Vec_reproduce
bash ~/2025/VLM2Vec_reproduce/experiments/public/train/train_retrieval.sh


echo "Job finished at: $(date)"
