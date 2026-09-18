#!/bin/bash
#SBATCH --job-name=cl_n2160_pairfix
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --account=CMPNS_inafts
#SBATCH --time=01:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/measure_%A_%a.out
#SBATCH --error=logs/measure_%A_%a.err

set -euo pipefail

BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_paired_fixed"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n2160_fullsky_paired_fixed"
mapfile -t REALIZATIONS < <(awk 'NF && $1 !~ /^#/' "${CAMPAIGN}/realizations.tsv")
read -r label _seed _paired <<<"${REALIZATIONS[${SLURM_ARRAY_TASK_ID}]}"
RUN="${BASE}/${label}"
cd "${RUN}"

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
set +u
source ~/miniforge3/bin/activate
conda activate geppetto-dev
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
	python "${REPO}/examples/measure_fullsky_angular_power.py" \
	--manifest "${RUN}/geppetto_reduced/painted_nfw_manifest.csv" \
	--params "${RUN}/params.txt" \
	--cosmology-table "${RUN}/pinocchio.000.cosmology.out" \
	--output "${RUN}/geppetto_reduced/fullsky_observed_spectra.npz"
