#!/bin/bash
#SBATCH --job-name=theory_l3870n4096_000
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --account=CMPNS_inafts
#SBATCH --time=4-00:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/theory_%j.out
#SBATCH --error=logs/theory_%j.err

set -eo pipefail

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

source ~/miniforge3/bin/activate
conda activate geppetto-dev
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
OUTDIR="${SLURM_SUBMIT_DIR}/geppetto_reduced/angular_power_validation"
mkdir -p "${OUTDIR}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
	python "${REPO}/examples/validate_pinocchio_angular_power.py" \
	--manifest "${SLURM_SUBMIT_DIR}/geppetto_reduced/painted_nfw_manifest.csv" \
	--params "${SLURM_SUBMIT_DIR}/params.txt" \
	--cosmology-table "${SLURM_SUBMIT_DIR}/pinocchio.000.cosmology.out" \
	--hmf-glob "${SLURM_SUBMIT_DIR}/pinocchio.*.000.mf.out" \
	--ell-exact-cap 512 \
	--limber-match-rtol 0.01 \
	--limber-match-width 20 \
	--exact-batch-size 112 \
	--exact-workers 112 \
	--exact-radial-order 512 \
	--exact-radial-tail-periods 256 \
	--finite-width-radial-order 256 \
	--finite-width-los-order 512 \
	--finite-width-tail-periods 40 \
	--output-dir "${OUTDIR}"
