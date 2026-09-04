#!/bin/bash
#SBATCH --job-name=geppetto_derivative_validation
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks=30
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=CMPNS_inafts
#SBATCH --output=logs/geppetto_derivative_validation_%j.out
#SBATCH --error=logs/geppetto_derivative_validation_%j.err

set -uo pipefail

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

set +u
source ~/miniforge3/bin/activate
conda activate geppetto-dev
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMPI_MCA_btl_tcp_if_include=ib0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/geppetto-matplotlib-${SLURM_JOB_ID}"

PARAMS="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/params.txt"
SHEETS="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.sheets.out"
PLC="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.plc.out"
MASSMAP_GLOB="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.massmap.seg0*.fits"
OUTDIR="${OUTDIR:-/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_derivative_validation}"
FD_RELATIVE_STEP="${FD_RELATIVE_STEP:-1e-4}"
GLOBAL_RTOL="${GLOBAL_RTOL:-1e-4}"
SHELL_RTOL="${SHELL_RTOL:-1e-3}"

mkdir -p "${OUTDIR}" "${MPLCONFIGDIR}"

echo "Derivative validation output: ${OUTDIR}"
echo "Finite-difference relative step: ${FD_RELATIVE_STEP}"
echo "Global/shell tolerances: ${GLOBAL_RTOL}/${SHELL_RTOL}"

set +e
srun --cpu-bind=cores python examples/paint_halo_particles_for_pinocchio_segment.py \
    --params "${PARAMS}" \
    --sheets "${SHEETS}" \
    --plc-catalog "${PLC}" \
    --mass-map-glob "${MASSMAP_GLOB}" \
    --output-dir "${OUTDIR}" \
    --mode derivatives-validate-profile \
    --nfw-overdensity 200 \
    --nfw-reference-density critical \
    --n-resolution 4 \
    --derivative-fd-relative-step "${FD_RELATIVE_STEP}" \
    --derivative-validation-global-rtol "${GLOBAL_RTOL}" \
    --derivative-validation-shell-rtol "${SHELL_RTOL}" \
    --jax-precision float64 \
    --mpi-plc-parts \
    --segment-workers 1
validation_status=$?
set -e

if [[ "${validation_status}" -eq 0 && -f "${OUTDIR}/painted_nfw_derivative_validation.csv" ]]; then
    python examples/plot_concentration_derivative_validation.py \
        --input-dir "${OUTDIR}" \
        --output-dir "${OUTDIR}"
fi

exit "${validation_status}"
