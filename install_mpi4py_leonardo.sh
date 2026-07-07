#!/bin/bash

# On Leonardo login node
module purge

# Load the same compiler/MPI stack you will use for runs.
# Use module spider/module avail to find the exact names on Leonardo.
module load gcc/12.2.0 
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

# Activate your conda env
source ~/miniforge3/bin/activate
conda activate geppetto-dev

# Sanity checks
which python
which mpicc
mpicc --version
mpicc -show  # or: mpicc --showme

# Build mpi4py from source against this mpicc
python -m pip uninstall -y mpi4py
MPICC="$(which mpicc)" python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

#Then verify:

python - <<'PY'
from mpi4py import MPI
print("mpi4py OK")
print("MPI library:", MPI.Get_library_version())
PY

#And test with multiple ranks:

srun -n 2 python - <<'PY'
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"rank {comm.rank}/{comm.size}")
PY

