#!/bin/bash
#SBATCH --job-name=pair_model_fullsky
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=CMPNS_inafts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=04:00:00
#SBATCH --array=0-12%4
#SBATCH --output=logs/pair_model_%A_%a.out
#SBATCH --error=logs/pair_model_%A_%a.err

set -euo pipefail

CAMPAIGN="${PAIR_MODEL_CAMPAIGN:-/leonardo_scratch/large/userexternal/tbatalha/Geppetto/outputs/fullsky_pair_model}"
BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160"
REFERENCE="${BASE}/000"
MANIFEST="${BASE}/fullsky_z0493_ensemble/seed001386/geppetto_reduced/painted_nfw_manifest.csv"
# Start the most expensive, nearest shells first.
segment=$((12 - SLURM_ARRAY_TASK_ID))
printf -v label '%03d' "${segment}"
batch=4
if (( segment >= 11 )); then
    batch=1
fi

module purge
set +u
source ~/miniforge3/bin/activate
conda activate geppetto-dev
set -u
export PYTHONPATH="${CAMPAIGN}/code/src:${CAMPAIGN}/dependencies${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
mkdir -p "${CAMPAIGN}/predictions"

# Keep the previous candidate's HMF, parameters and linear prediction fixed.
# Full-sky IDs 0..12 correspond to parent IDs 16..28; seed 737 preserves
# the previous orientation draws (721 + parent segment).
srun --cpu-bind=cores python "${CAMPAIGN}/code/predict_painting_matched_power.py" \
    --params "${REFERENCE}/params.txt" \
    --cosmology-table "${REFERENCE}/pinocchio.000.cosmology.out" \
    --hmf-glob "${REFERENCE}/pinocchio.*.000.mf.out" \
    --manifest "${MANIFEST}" \
    --linear-reference "${CAMPAIGN}/linear_reference.npz" \
    --segments "${segment}" --ell-max 2000 \
    --moment-backend auto --mass-batch-size "${batch}" \
    --orientations 16 --seed 737 --threads "${SLURM_CPUS_PER_TASK}" \
    --output "${CAMPAIGN}/predictions/shell_${label}.npz"
