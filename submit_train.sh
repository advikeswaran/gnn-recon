#!/bin/bash
#PBS -N recon_train
#PBS -A P93300301
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=1:mem=256GB
#PBS -l walltime=12:00:00
#PBS -l job_priority=premium
#PBS -o /glade/derecho/scratch/advike/graphcast_recon/logs/train.out
#PBS -e /glade/derecho/scratch/advike/graphcast_recon/logs/train.err
#PBS -j oe

module load cuda/12.9.0
module load conda/latest
source $(conda info --base)/etc/profile.d/conda.sh
conda activate graphcast

cd /glade/derecho/scratch/advike/graphcast_recon

python train_head.py \
    --epochs 5 \
    --lr 1e-3 \
    --hidden 128 \
    --t2t-rounds 6 \
    --dry-run
