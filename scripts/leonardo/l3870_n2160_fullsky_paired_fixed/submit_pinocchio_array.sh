#!/bin/bash
#SBATCH --job-name=pino_n2160_pairfix
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_bprod
#SBATCH --nodes=24
#SBATCH --ntasks=720
#SBATCH --ntasks-per-node=30
#SBATCH --cpus-per-task=3
#SBATCH --account=CMPNS_inafts
#SBATCH --time=04:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/pinocchio_%A_%a.out
#SBATCH --error=logs/pinocchio_%A_%a.err

set -euo pipefail

BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_paired_fixed"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n2160_fullsky_paired_fixed"
mapfile -t REALIZATIONS < <(awk 'NF && $1 !~ /^#/' "${CAMPAIGN}/realizations.tsv")
read -r label seed paired <<<"${REALIZATIONS[${SLURM_ARRAY_TASK_ID}]}"
cd "${BASE}/${label}"

module purge
module load gcc/12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0-spack0.22
module load gsl/2.7.1--gcc--12.2.0-spack0.22
module load hdf5/1.14.3--openmpi--4.1.6--gcc--12.2.0-spack0.22
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/leonardo_scratch/large/userexternal/tbatalha/Pinocchio/dep/lib:/leonardo_scratch/large/userexternal/tbatalha/Pinocchio/dep/Healpix_3.83/lib"

echo "Realization: ${label}; seed: ${seed}; paired: ${paired}; fixed: 1"
echo "PINOCCHIO binary: $(sha256sum pinocchio.x)"
srun --ntasks=720 --cpus-per-task=3 --cpu-bind=cores ./pinocchio.x params.txt

plc_parts=$(find . -maxdepth 1 -type f -name 'pinocchio.000.plc.out.*' | wc -l)
mass_maps=$(find . -maxdepth 1 -type f -name 'pinocchio.000.massmap.seg*.fits' | wc -l)
if ((plc_parts != 30 || mass_maps != 13)); then
	echo "Unexpected outputs: PLC parts=${plc_parts}, mass maps=${mass_maps}" >&2
	exit 2
fi
rm -f pinocchio.*.000.catalog.out*
