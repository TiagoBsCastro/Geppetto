#!/bin/bash
#SBATCH --job-name=geppetto_galaxy_input
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=CMPNS_inafts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --output=prepare_%j.out
#SBATCH --error=prepare_%j.err
set -euo pipefail

INPUT=${INPUT:-/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N4096/000}
PYTHON=${PYTHON:-/leonardo/home/userexternal/tbatalha/miniforge3/envs/geppetto-dev/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" examples/prepare_pinocchio_galaxy_input.py \
    --params "${INPUT}/params.txt" \
    --cosmology-table "${INPUT}/pinocchio.000.cosmology.out" \
    --plc-catalog "${INPUT}/pinocchio.000.plc.out" \
    --z-min 0.04 --z-max 0.33 --output halos_z004_z033.hdf5
