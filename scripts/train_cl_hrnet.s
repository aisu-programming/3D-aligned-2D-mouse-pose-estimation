#!/bin/bash
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=cl_hr
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurm_%j_cl_hrnet.out
#SBATCH --error=slurm_%j_cl_hrnet.err

cd /scratch//CLIP-pose-estimation
source .venv/bin/activate
python -m train.train_cl_hrnet.py
