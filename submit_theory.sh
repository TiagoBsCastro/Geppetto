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
# OpenMP binding can pin Python before its process pool is spawned.
unset OMP_PLACES
export OMP_PROC_BIND=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE=/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160
CASE="${SCHEMA6_CASE:-cutsky}"
ARGS=()
if [[ "${CASE}" == cutsky ]]; then
    OUTDIR="${OUTDIR:-${BASE}/000/geppetto_reduced/angular_power_validation_schema6}"
    ARGS+=(--manifest "${BASE}/000/geppetto_reduced/painted_nfw_manifest.csv")
    ARGS+=(--hmf-glob "${BASE}/000/pinocchio.*.000.mf.out")
    ARGS+=(--measurement-dir "${BASE}/000/geppetto_reduced/angular_power_validation_schema5")
elif [[ "${CASE}" == ensemble ]]; then
    ENSEMBLE="${BASE}/fullsky_z0493_ensemble"
    OUTDIR="${OUTDIR:-${ENSEMBLE}/angular_power_fullsky_ensemble_schema6}"
    ARGS+=(--manifest "${ENSEMBLE}/seed001386/geppetto_reduced/painted_nfw_manifest.csv")
    ARGS+=(--fullsky-ensemble-root "${ENSEMBLE}")
    for seed in "${ENSEMBLE}"/seed*; do
        [[ -d "${seed}" ]] || continue
        ARGS+=(--hmf-glob "${seed}/*.mf.out")
    done
else
    printf 'Unknown SCHEMA6_CASE: %s\n' "${CASE}" >&2
    exit 1
fi
mkdir -p "${OUTDIR}"

python - <<'PY'
from cctoolkit.bias import bias_correction_PBS

print("CCToolkit bias correction:", bias_correction_PBS.__module__)
PY

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-112}" --cpu-bind=none \
	python examples/validate_normalized_halo_model.py \
	--params "${BASE}/000/params.txt" \
	--cosmology-table "${BASE}/000/pinocchio.000.cosmology.out" \
	"${ARGS[@]}" \
	--ell-exact-cap 512 \
	--limber-match-rtol 0.01 \
	--limber-match-width 20 \
	--exact-batch-size 112 \
	--exact-workers 112 \
	--output-dir "${OUTDIR}"

python examples/plot_schema6_halo_model.py --input-dir "${OUTDIR}" --output-dir "${OUTDIR}/figures"
if [[ "${CASE}" == cutsky ]]; then
    python examples/plot_angular_power_validation.py \
        --input-dir "${OUTDIR}" --output-dir "${OUTDIR}/figures"
else
    OBS=()
    for seed in "${ENSEMBLE}"/seed*; do
        [[ -d "${seed}" ]] || continue
        label="${seed##*/seed}"
        OBS+=(--seed "$((10#${label}))" --observed-cache "${seed}/geppetto_reduced/fullsky_observed_spectra.npz")
    done
    for model in standard-halo-model standard-halo-model-uncompensated; do
        mkdir -p "${OUTDIR}/${model}"
        python examples/aggregate_angular_power_ensemble.py \
            --theory-dir "${OUTDIR}" "${OBS[@]}" --theory-model "${model}" \
            --require-scale-dependent-camb --output-dir "${OUTDIR}/${model}"
    done
fi
