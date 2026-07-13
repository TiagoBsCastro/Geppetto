#!/bin/bash
#SBATCH --job-name=geppetto_l3870n2160_000
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --account=CMPNS_inafts
#SBATCH --output=logs/geppetto_%j.out
#SBATCH --error=logs/geppetto_%j.err

set -eo pipefail

# The run has 30 split PLC files, so it requires 30 MPI ranks. Three segment
# workers per rank use 90 of Leonardo DCGP's 112 physical cores. Four workers
# per rank would require 120 cores and therefore cannot fit on one node.

module purge

# Load the same compiler/MPI stack used when installing mpi4py.
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

# Conda activation hooks may reference unset variables.
set +u
source ~/miniforge3/bin/activate
conda activate geppetto-dev
set -u

echo "Python: $(which python)"
echo "mpicc: $(which mpicc)"
python - <<'PY'
from mpi4py import MPI

print("mpi4py MPI version:", MPI.Get_version())
print(MPI.Get_library_version())
PY

# Avoid each segment worker spawning extra threaded BLAS/OpenMP work.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export PARAMS="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/params.txt"
export SHEETS="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.sheets.out"
export PLC="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.plc.out"
export MASSMAP_GLOB="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.massmap.seg0*.fits"
export OUTDIR="${OUTDIR:-/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced}"
export GEPPETTO_MODE="${GEPPETTO_MODE:-derivatives-profile}"
export SEGMENT_WORKERS="${SEGMENT_WORKERS:-$SLURM_CPUS_PER_TASK}"
export STENCIL_QUERY_MODE="${STENCIL_QUERY_MODE:-center}"

if ((SEGMENT_WORKERS < 1 || SEGMENT_WORKERS > SLURM_CPUS_PER_TASK)); then
	echo "SEGMENT_WORKERS must be between 1 and SLURM_CPUS_PER_TASK" >&2
	exit 2
fi
if [[ "${STENCIL_QUERY_MODE}" != "center" && "${STENCIL_QUERY_MODE}" != "inclusive" ]]; then
	echo "STENCIL_QUERY_MODE must be center or inclusive" >&2
	exit 2
fi

mkdir -p "${OUTDIR}"
echo "GEPPETTO mode: ${GEPPETTO_MODE}"
echo "Segment workers: ${SEGMENT_WORKERS}"
echo "Stencil query mode: ${STENCIL_QUERY_MODE}"
echo "Output directory: ${OUTDIR}"

srun --cpu-bind=cores python examples/paint_halo_particles_for_pinocchio_segment.py \
	--params "${PARAMS}" \
	--sheets "${SHEETS}" \
	--plc-catalog "${PLC}" \
	--mass-map-glob "${MASSMAP_GLOB}" \
	--output-dir "${OUTDIR}" \
	--mode "${GEPPETTO_MODE}" \
	--mpi-plc-parts \
	--segment-workers "${SEGMENT_WORKERS}" \
	--stencil-query-mode "${STENCIL_QUERY_MODE}"
