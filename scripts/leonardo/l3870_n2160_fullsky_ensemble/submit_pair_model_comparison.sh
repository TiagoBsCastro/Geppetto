#!/bin/bash
#SBATCH --job-name=compare_pair_fullsky
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=CMPNS_inafts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/pair_compare_%j.out
#SBATCH --error=logs/pair_compare_%j.err

set -euo pipefail
CAMPAIGN="${PAIR_MODEL_CAMPAIGN:-/leonardo_scratch/large/userexternal/tbatalha/Geppetto/outputs/fullsky_pair_model}"
BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160"
module purge
set +u
source ~/miniforge3/bin/activate
conda activate geppetto-dev
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pair-model-matplotlib-${SLURM_JOB_ID}"
python "${CAMPAIGN}/code/compare_fullsky_pair_model.py" \
    --prediction-glob "${CAMPAIGN}/predictions/shell_*.npz" \
    --ensemble-root "${BASE}/fullsky_z0493_ensemble" \
    --scheme5 "${BASE}/000/geppetto_reduced/angular_power_validation_schema5" \
    --scheme6 "${BASE}/fullsky_z0493_ensemble/angular_power_fullsky_ensemble_schema6" \
    --output-dir "${CAMPAIGN}/comparison"
