#!/bin/bash
#SBATCH --job-name=quick_l3870_resolution
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --account=CMPNS_inafts
#SBATCH --time=02:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/quick_compare_%j.out
#SBATCH --error=logs/quick_compare_%j.err

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"

REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
LOW="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/angular_power_validation"
OUTDIR="${SLURM_SUBMIT_DIR}/geppetto_reduced/quick_low_theory_comparison"
mkdir -p "${OUTDIR}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
	python "${REPO}/examples/compare_observed_resolutions_to_low_theory.py" \
	--low-dir "${LOW}" \
	--high-manifest "${SLURM_SUBMIT_DIR}/geppetto_reduced/painted_nfw_manifest.csv" \
	--output-dir "${OUTDIR}" \
	--representative-segment 23
