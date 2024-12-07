#!/bin/bash
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=u_w
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurm_%j_train_unet_wandb.out
#SBATCH --error=slurm_%j_train_unet_wandb.err

cd /scratch//CLIP-pose-estimation
source .venv/bin/activate
python -m train.train_unet_wandb.py
