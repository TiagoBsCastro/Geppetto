#!/bin/bash
#SBATCH --job-name=geppetto_n2160_pairfix
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=3
#SBATCH --account=CMPNS_inafts
#SBATCH --time=04:00:00
#SBATCH --mem=494000M
#SBATCH --gres=tmpfs:10g
#SBATCH --exclusive
#SBATCH --output=logs/geppetto_%A_%a.out
#SBATCH --error=logs/geppetto_%A_%a.err

set -euo pipefail

BASE="/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_paired_fixed"
REPO="${GEPPETTO_REPO:-${HOME}/scratch/Geppetto}"
CAMPAIGN="${REPO}/scripts/leonardo/l3870_n2160_fullsky_paired_fixed"
mapfile -t REALIZATIONS < <(awk 'NF && $1 !~ /^#/' "${CAMPAIGN}/realizations.tsv")
read -r label _seed _paired <<<"${REALIZATIONS[${SLURM_ARRAY_TASK_ID}]}"
RUN="${BASE}/${label}"
OUTDIR="${RUN}/geppetto_reduced"
cd "${RUN}"

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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUTDIR}"
srun --cpu-bind=cores python "${REPO}/examples/paint_halo_particles_for_pinocchio_segment.py" \
	--params "${RUN}/params.txt" \
	--sheets "${RUN}/pinocchio.000.sheets.out" \
	--plc-catalog "${RUN}/pinocchio.000.plc.out" \
	--mass-map-glob "${RUN}/pinocchio.000.massmap.seg*.fits" \
	--output-dir "${OUTDIR}" \
	--mode profile \
	--nfw-overdensity 200 \
	--nfw-reference-density critical \
	--n-resolution 4 \
	--mpi-plc-parts \
	--segment-workers "${SLURM_CPUS_PER_TASK}"

test "$(find "${OUTDIR}" -maxdepth 1 -type f -name 'painted_nfw.seg*.npz' | wc -l)" -eq 13
test -f "${OUTDIR}/painted_nfw_manifest.csv"
