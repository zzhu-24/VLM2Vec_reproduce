#!/bin/bash
#SBATCH --job-name=eval               # Name of your job
#SBATCH --output=./slurm_results/%x/%j.out
#SBATCH --error=./slurm_results/%x/%j.err
#SBATCH --partition=3090              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=30G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vlm2vec
cd /home/infres/zzhu-24/PRIM/VLM2Vec
bash /home/infres/zzhu-24/PRIM/VLM2Vec/experiments/public/eval/eval_after_train.sh


echo "Job finished at: $(date)"
