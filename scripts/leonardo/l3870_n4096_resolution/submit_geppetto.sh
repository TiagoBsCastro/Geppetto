#!/bin/bash
#SBATCH --job-name=geppetto_l3870n4096_000
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --account=CMPNS_inafts
#SBATCH --time=12:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/geppetto_%j.out
#SBATCH --error=logs/geppetto_%j.err

set -eo pipefail

module purge
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

source ~/miniforge3/bin/activate
conda activate geppetto-dev
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SRUN_CPUS_PER_TASK=1

REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
OUTDIR="${SLURM_SUBMIT_DIR}/geppetto_reduced"
mapfile -t PLC_PARTS < <(find . -maxdepth 1 -type f -name 'pinocchio.000.plc.out.*' | sort -V)
if ((${#PLC_PARTS[@]} != SLURM_NTASKS)); then
	echo "Expected ${SLURM_NTASKS} PLC parts, found ${#PLC_PARTS[@]}" >&2
	exit 2
fi
mkdir -p "${OUTDIR}"

srun --cpu-bind=cores python "${REPO}/examples/paint_halo_particles_for_pinocchio_segment.py" \
	--params "${SLURM_SUBMIT_DIR}/params.txt" \
	--sheets "${SLURM_SUBMIT_DIR}/pinocchio.000.sheets.out" \
	--plc-catalog "${SLURM_SUBMIT_DIR}/pinocchio.000.plc.out" \
	--mass-map-glob "${SLURM_SUBMIT_DIR}/pinocchio.000.massmap.seg0*.fits" \
	--output-dir "${OUTDIR}" \
	--mode profile \
	--nfw-overdensity 200 \
	--nfw-reference-density critical \
	--n-resolution 4 \
	--mpi-plc-parts \
	--segment-workers 1
