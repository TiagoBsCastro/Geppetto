#!/bin/bash
#SBATCH --job-name=compare_l3870_resolution
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=CMPNS_inafts
#SBATCH --time=01:00:00
#SBATCH --mem=16000M
#SBATCH --gres=tmpfs:10g
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

set -eo pipefail

source ~/miniforge3/bin/activate
conda activate geppetto-dev
cd "${SLURM_SUBMIT_DIR}"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
LOW="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/angular_power_validation"
HIGH="${SLURM_SUBMIT_DIR}/geppetto_reduced/angular_power_validation"
OUTDIR="${SLURM_SUBMIT_DIR}/geppetto_reduced/resolution_comparison"

python "${REPO}/examples/compare_angular_power_resolutions.py" \
	--low-dir "${LOW}" \
	--high-dir "${HIGH}" \
	--output-dir "${OUTDIR}" \
	--representative-segment 23
