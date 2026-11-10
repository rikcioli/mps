#!/bin/bash
#SBATCH --job-name=inv_parallel
#SBATCH --output=inv_parallel_%j.log
#SBATCH --error=inv_parallel_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

julia mps/driver_parallel.jl -continue /local/cioli/data/xxz/Jz2.5 -T 3_28
