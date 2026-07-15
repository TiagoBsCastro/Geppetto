#!/bin/bash
#SBATCH --job-name=geppetto_theory
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --account=CMPNS_inafts
#SBATCH --output=logs/geppetto_theory_%j.out
#SBATCH --error=logs/geppetto_theory_%j.err

set -eo pipefail

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

source ~/miniforge3/bin/activate
conda activate geppetto-dev
cd ~/scratch/Geppetto

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python examples/validate_pinocchio_angular_power.py \
	--manifest /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/painted_nfw_manifest.csv \
	--params /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/params.txt \
	--cosmology-table /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.cosmology.out \
	--hmf-glob '/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.*.000.mf.out' \
	--ell-exact-cap 512 \
	--limber-match-rtol 0.01 \
	--limber-match-width 20 \
	--output-dir /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/angular_power_validation
