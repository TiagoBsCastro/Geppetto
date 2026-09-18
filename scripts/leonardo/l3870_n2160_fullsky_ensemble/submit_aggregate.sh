#!/bin/bash
#SBATCH --job-name=aggregate_n2160_fullsky
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=CMPNS_inafts
#SBATCH --time=01:00:00
#SBATCH --mem=32000M
#SBATCH --gres=tmpfs:10g
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -euo pipefail

BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_ensemble"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n2160_fullsky_ensemble"
THEORY="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced/angular_power_validation_schema5"
OUTDIR="${BASE}/ensemble_results"
mapfile -t SEEDS <"${CAMPAIGN}/seeds.txt"

module purge
set +u
source ~/miniforge3/bin/activate
conda activate geppetto-dev
set -u
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

args=()
for seed in "${SEEDS[@]}"; do
	printf -v label 'seed%06d' "${seed}"
	args+=(
		--seed "${seed}"
		--observed-cache "${BASE}/${label}/geppetto_reduced/fullsky_observed_spectra.npz"
	)
done

python "${REPO}/examples/aggregate_angular_power_ensemble.py" \
	--theory-dir "${THEORY}" \
	--theory-model linear-baseline \
	--require-scale-dependent-camb \
	--output-dir "${OUTDIR}" \
	"${args[@]}"
