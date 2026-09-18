#!/bin/bash
#SBATCH --job-name=pinocchio_l3870n4096_000
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --account=CMPNS_inafts
#SBATCH --time=04:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/pinocchio_%j.out
#SBATCH --error=logs/pinocchio_%j.err

set -eo pipefail

module purge
module load gcc/12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0-spack0.22
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/leonardo_scratch/large/userexternal/tbatalha/Pinocchio/dep/lib:/leonardo_scratch/large/userexternal/tbatalha/Pinocchio/dep/Healpix_3.83/lib"

test -x pinocchio.x
test -f params.txt
test -f outputs
test -d CambFiles
echo "PINOCCHIO binary: $(sha256sum pinocchio.x)"
echo "MPI ranks: ${SLURM_NTASKS}; OpenMP threads per rank: ${OMP_NUM_THREADS}"

srun --cpu-bind=cores ./pinocchio.x params.txt
