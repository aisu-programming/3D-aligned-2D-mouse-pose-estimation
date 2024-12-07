#!/bin/bash
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=cl_u
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurm_%j_cl_u.out
#SBATCH --error=slurm_%j_cl_u.err

cd /scratch//CLIP-pose-estimation
source .venv/bin/activate
python -m train.train_cl_unet.py
