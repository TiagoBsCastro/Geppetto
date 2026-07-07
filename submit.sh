#!/bin/bash
#SBATCH --job-name=geppetto_l3870n2160_000
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=10
#SBATCH --ntasks=30
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=29
#SBATCH --time=04:00:00
#SBATCH --account=CMPNS_inafts
#SBATCH --output=logs/geppetto_%j.out
#SBATCH --error=logs/geppetto_%j.err

set -eo pipefail

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

export PARAMS="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/params.txt"
export SHEETS="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.sheets.out"
export PLC="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.plc.out"
export MASSMAP_GLOB="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/pinocchio.000.massmap.seg0*.fits"
export OUTDIR="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/000/geppetto_reduced"

mkdir -p "${OUTDIR}"

srun --cpu-bind=cores python examples/paint_halo_particles_for_pinocchio_segment.py \
	--params "${PARAMS}" \
	--sheets "${SHEETS}" \
	--plc-catalog "${PLC}" \
	--mass-map-glob "${MASSMAP_GLOB}" \
	--output-dir "${OUTDIR}" \
	--mode derivatives \
	--mpi-plc-parts \
	--mpi-output-mode reduce \
	--segment-workers 1
