#!/bin/bash 

#SBATCH --job-name=alphafold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --gres=gpu

module purge

/vast/wang/alphafold-20211025/run-alphafold.py \
    --fasta_paths=T1050.fasta \
    --output_dir=output \
    --max_template_date=2020-05-14 \
    --preset=full_dbs



