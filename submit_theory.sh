#!/bin/bash
#SBATCH --job-name=geppetto_theory
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --qos=dcgp_qos_lprod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=494000MB
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
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
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-112}" --cpu-bind=none \
	python examples/validate_pinocchio_angular_power.py \
	--manifest /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/painted_nfw_manifest.csv \
	--params /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/params.txt \
	--cosmology-table /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.cosmology.out \
	--hmf-glob '/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.*.000.mf.out' \
	--ell-exact-cap 512 \
	--limber-match-rtol 0.01 \
	--limber-match-width 20 \
	--exact-batch-size 112 \
	--exact-workers 112 \
	--exact-radial-order 512 \
	--exact-radial-tail-periods 256 \
	--output-dir /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/angular_power_validation
