#!/bin/bash
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_hrnet
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurm_%j_train_hrnet.out
#SBATCH --error=slurm_%j_train_hrnet.err

cd /scratch//CLIP-pose-estimation
source .venv/bin/activate
python -m train.train_hrnet.py
