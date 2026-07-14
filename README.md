# GEPPETTO

**GEPPETTO** is a differentiable JAX one-halo matter/profile painter for
PINOCCHIO halo catalogues.

PINOCCHIO provides the halo catalogue, past-light-cone geometry, and
large-scale/two-halo HEALPix mass maps. GEPPETTO supplies a local one-halo layer
that can be painted around catalogued haloes, differentiated with respect to
profile and concentration parameters, and compared against theoretical
predictions.

Current status: **active research prototype**. The repository has working NFW
box/lightcone painters, PINOCCHIO readers, sparse HEALPix painting, an
all-segments PINOCCHIO calibration script, and map-level derivatives for
concentration-mass parameters. APIs and physical models are still evolving.

## What GEPPETTO Does

- Paints NFW one-halo density fields in comoving boxes and projected
  lightcones.
- Reads PINOCCHIO snapshot catalogues, PLC catalogues, mass-sheet tables,
  mass-map FITS files, Hubble tables, mass functions, and `nz` tables.
- Produces PINOCCHIO-compatible one-halo particle-count-equivalent HEALPix maps.
- Builds differentiable sparse PLC maps from fixed halo-pixel stencils.
- Computes map-level derivatives with respect to concentration amplitude, mass
  slope, and redshift slope for c-M calibration workflows.
- Keeps HEALPix indexing, file I/O, and other discrete geometry outside the JAX
  kernels.

GEPPETTO does **not** replace PINOCCHIO lightcone generation and does not merge
one-halo maps with PINOCCHIO's two-halo maps automatically.

## Angular-Power Scientific Validation

GEPPETTO includes a linear-plus-one-halo prediction for the angular spectra of
the combined PINOCCHIO uncollapsed-particle and painted-halo count maps:

```text
P_mm(k,z) = P_linear(k,z) + P_1h(k,z)
```

The linear spectrum, growth, distances, and background density are read from
PINOCCHIO's `*.cosmology.out`. The one-halo term uses measured `*.mf.out` mass
functions and the same concentration relation and mass definition recorded in
`painted_nfw_manifest.csv`. Exact spherical-Bessel projection is used through
`ell=100`; Limber projection is used above that and for the one-halo term.

Run the map comparison after painting all segments:

```bash
python examples/validate_pinocchio_angular_power.py \
  --manifest /path/to/painted_nfw_manifest.csv \
  --cosmology-table /path/to/pinocchio.RUN.cosmology.out \
  --hmf-glob '/path/to/pinocchio.*.RUN.mf.out' \
  --output-dir /path/to/angular-power-validation
```

The command writes lean NPZ and CSV products containing measured and predicted
spectra, particle shot noise, shell weights, resolved HMF mass fractions, and
the low-k one-halo/linear ratio. It does not write another copy of any map.
See [docs/angular_power_validation.md](docs/angular_power_validation.md) for
the equations, units, estimator limitations, and output definitions.

## Installation

GEPPETTO is currently installed from the GitHub repository. The recommended
research/development setup uses Miniforge or Mambaforge plus an editable pip
install:

```bash
git clone git@github.com:TiagoBsCastro/Geppetto.git
cd Geppetto

mamba create -n geppetto-dev python=3.12
mamba activate geppetto-dev

python -m pip install -e '.[io,theory,dev]'
```

Installation extras:

- `python -m pip install -e .` installs the differentiable JAX/NumPy core.
- `python -m pip install -e '.[io]'` adds `astropy`, `healpy`, and `h5py` for
  FITS, HEALPix, HDF5, and PINOCCHIO reader workflows.
- `python -m pip install -e '.[dev]'` adds `pytest`, `ruff`, and `mypy`.
- `python -m pip install -e '.[theory]'` adds SciPy for exact low-multipole
  spherical-Bessel projection.
- `python -m pip install -e '.[io,theory,dev]'` is recommended for running all
  examples and tests.

Verify the installation:

```bash
python -c "import geppetto; print(geppetto.__version__)"
pytest
ruff check .
```

For GPU use, install the JAX build appropriate for your CUDA stack before
installing GEPPETTO. Follow the official JAX installation selector rather than
pinning a CUDA command from this README.

## Quick Start: Python Painters

### Comoving Box

```python
import jax.numpy as jnp

from geppetto import Cosmology, HaloCatalog, duffy08_all_200c, paint_box_density_grid

catalog = HaloCatalog(
    position=jnp.array([[50.0, 50.0, 50.0]]),  # comoving Mpc/h
    mass=jnp.array([1.0e14]),                  # Msun/h
    redshift=jnp.array([0.0]),
)

grid = paint_box_density_grid(
    catalog,
    box_size=100.0,
    nmesh=32,
    cosmology=Cosmology(omega_m=0.315, h=0.674),
    concentration_params=duffy08_all_200c(),
    periodic=True,
)

print(grid.shape)  # (32, 32, 32)
```

### Lightcone Surface Density

GEPPETTO's differentiable core receives fixed pixel unit vectors, not HEALPix
indices. Convert HEALPix pixels outside the core, for example with
`geppetto.io.healpix_pixel_unit_vectors`.

```python
import jax.numpy as jnp

from geppetto import LightconeHaloCatalog, paint_lightcone_surface_density

pixel_unit_vectors = jnp.array([
    [1.0, 0.0, 0.0],
    [0.999, 0.045, 0.0],
])

catalog = LightconeHaloCatalog(
    unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
    chi=jnp.array([1000.0]),      # comoving Mpc/h
    mass=jnp.array([1.0e14]),     # Msun/h
    redshift=jnp.array([0.3]),
)

sigma = paint_lightcone_surface_density(pixel_unit_vectors, catalog)
print(sigma)
```

### Sparse PLC Painting

For large angular maps, build a fixed halo-pixel stencil outside JAX and paint
only retained local pairs:

```python
from geppetto import paint_lightcone_surface_density_sparse
from geppetto.io import build_lightcone_sparse_stencil_bruteforce

stencil = build_lightcone_sparse_stencil_bruteforce(
    pixel_unit_vectors,
    catalog,
    rmax_mpc_h=5.0,
)

sigma_sparse = paint_lightcone_surface_density_sparse(stencil, catalog)
```

`build_lightcone_sparse_stencil_bruteforce` materializes an `n_pix * n_halo`
separation matrix and is intended for tests, examples, and small maps. The
PINOCCHIO calibration script below uses a HEALPix-local sparse stencil builder
for segment painting. `LightconeSparseStencil.pair_weight` optionally applies a
dimensionless weight to each retained pair; omitted weights are one. Zero
weights are useful for shape padding because padded pairs then contribute
exactly zero even when they repeat valid pixel and halo indices.

## PINOCCHIO c-M Calibration Pipeline

The main PINOCCHIO workflow is:

```text
PINOCCHIO PLC halo catalogue
    -> NFW one-halo particle-count map
    -> optional map-level derivatives wrt concentration-mass parameters
```

The command-line entrypoint is:

```text
examples/paint_halo_particles_for_pinocchio_segment.py
```

It always writes an NFW painted map in particle-count-equivalent units. In
derivative modes it also saves:

```text
d_nfw_particle_counts_d_concentration_amplitude
d_nfw_particle_counts_d_concentration_mass_slope
d_nfw_particle_counts_d_concentration_redshift_slope
```

These derivative maps have the same compact pixel shape and ordering as the
corresponding PINOCCHIO mass-map segment.

### Single Segment

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params examples/pinocchio_geppetto_case/parameter_file \
  --sheets examples/pinocchio_geppetto_case/pinocchio.example.sheets.out \
  --mass-map examples/pinocchio_geppetto_case/pinocchio.example.massmap.seg001.fits \
  --plc-catalog examples/pinocchio_geppetto_case/pinocchio.example.plc.out \
  --sheet-index 1 \
  --output examples/pinocchio_geppetto_case/halo_particles.seg001.paint.npz \
  --nfw-overdensity 200 \
  --nfw-reference-density critical
```

Add map-level derivatives:

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params examples/pinocchio_geppetto_case/parameter_file \
  --sheets examples/pinocchio_geppetto_case/pinocchio.example.sheets.out \
  --mass-map examples/pinocchio_geppetto_case/pinocchio.example.massmap.seg001.fits \
  --plc-catalog examples/pinocchio_geppetto_case/pinocchio.example.plc.out \
  --sheet-index 1 \
  --output examples/pinocchio_geppetto_case/halo_particles.seg001.derivs.npz \
  --nfw-overdensity 200 \
  --nfw-reference-density critical \
  --mode derivatives
```

### All Segments

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params examples/pinocchio_geppetto_case/parameter_file \
  --sheets examples/pinocchio_geppetto_case/pinocchio.example.sheets.out \
  --plc-catalog examples/pinocchio_geppetto_case/pinocchio.example.plc.out \
  --mass-map-glob "examples/pinocchio_geppetto_case/pinocchio.example.massmap.seg*.fits" \
  --output-dir examples/pinocchio_geppetto_case/painted_nfw \
  --nfw-overdensity 200 \
  --nfw-reference-density critical \
  --mode derivatives
```

All-segments mode writes one compressed segment-local NPZ per input mass-map
segment and one provenance manifest:

```text
painted_nfw.seg000.npz
painted_nfw.seg001.npz
painted_nfw_manifest.csv
```

Every NPZ contains only `nfw_particle_counts`; derivative modes add the three
derivative arrays listed above. Pixel identifiers and HEALPix metadata are not
copied: array row `i` corresponds to row `i` in the original PINOCCHIO mass-map
FITS file named by `mass_map_path` in the manifest. The manifest records input
paths, segment bounds, numerical precision, physical painter parameters, MPI
and worker configuration, the Git commit, and compact scientific diagnostics.
These include selected halo count and mass, expected and assigned global count,
retained and outside-compact count, adaptive branch counts, profile-sample
counts, and supersampling levels. Timing diagnostics stay in profile logs. The
script does not produce a merged global light-cone map.

The NFW halo-mass interpretation is required rather than implicit. Select
exactly one of:

```text
--nfw-overdensity DELTA
--nfw-virial-overdensity
--nfw-overdensity-by-segment path/to/delta.npy
```

All three require `--nfw-reference-density critical` or `mean`. A constant
`DELTA` may be any finite positive spherical overdensity. The Bryan--Norman
mode evaluates `Delta_vir,c(z)` per halo; its mean-density convention uses
`Delta_vir,m(z) = Delta_vir,c(z) / Omega_m(z)` and therefore preserves the same
physical threshold. The `.npy` mode requires a one-dimensional positive array
with exactly one value per sheet row. Catalogue masses are interpreted directly
in the selected definition; no mass conversion is applied, and this fact is
recorded in the manifest.

All-segments mode requires one contiguous, sheet-valid range. Every segment is
half-open except the actual final sheet, `len(sheets) - 1`, which is inclusive.
A partial glob therefore remains half-open at its highest segment and can be
continued in another batch without double-selecting a boundary halo.
Radial segment membership is defined by the halo centre. Once selected, the
halo's complete projected profile is painted into that segment; profiles are
not clipped or divided at radial sheet boundaries.

Advanced parallel mode is opt-in. `--mpi-plc-parts` uses one MPI rank per split
PLC catalogue part named like `pinocchio.RUN.plc.out.0`,
`pinocchio.RUN.plc.out.1`, and so on; the MPI world size must equal the number
of discovered parts. Each rank paints only its local halo subset, then rank 0
reduces and writes final segment outputs without temporary per-rank map files.
Rank-local failures report the rank and segment and call `MPI.Comm.Abort`, so
other ranks do not remain blocked in a collective.

The production painter projects the three-dimensional NFW density only through
the sphere `r <= R_delta`. Its exact finite line-of-sight surface density has
hard projected support at `R_delta`; no taper or support extension is applied.
The concentration-independent angular support is
`2 asin(min(1, R_delta / (2 chi)))`.

Angular assignment is adaptive. Halos below `--theta-resolution-rad` use NGP
without a disc query; the default threshold is half the maximum native HEALPix
pixel radius. Halos between that threshold and `--n-resolution` times the
area-equivalent pixel scale are queried on a directly refined NESTED grid.
Larger halos use native RING pixel centers. Refined children are mapped to
native NESTED parents and then converted to RING rows.

Resolved profile weights are normalized inside JAX over complete global
HEALPix support before compact-map filtering. Every halo therefore assigns its
full catalogue mass globally, while a compact segment may retain less when a
profile crosses its angular boundary. Concentration JVPs include the derivative
of this normalization, so their complete global sums vanish. Fixed sample
geometry is padded for JIT reuse and evaluated in bounded chunks controlled by
the hidden numerical option `--nfw-sample-chunk-size`.

`--segment-workers N` is a bounded prefetch window over mass-map segments, not
an `N`-fold speedup for one segment. The main thread reduces segments in
deterministic order while worker threads can compute later segments, including
while the main thread waits for other ranks in MPI. The per-halo
`healpy.query_disc` loop remains Python/GIL-bound and is not directly
parallelized by this option. With one worker, MPI mode streams one segment at a
time. Memory grows approximately linearly with the worker count: for the
16,560,128-pixel float64 derivative workflow, the four retained output arrays
occupy about 0.49 GiB per rank before temporary pixel-index, stencil, and JAX
allocations. Compressed NPZ size is data-dependent.

The checked-in Leonardo DCGP job uses 30 ranks on one 112-core node, three CPUs
per rank, and three segment workers. The CPU request bounds concurrent worker
activity and prevents oversubscription; it is not an estimate of linear
speedup. Testing four workers per rank should also raise `--cpus-per-task`, so
30 such tasks no longer fit on one 112-core node. Use `derivatives-profile` to
print root-only per-segment and all-segment min/mean/max rank timings for
compute, prefetched-result waiting, and MPI reduction; normal `derivatives`
mode performs no timing gather.

```bash
mpiexec -n 4 python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params path/to/parameter_file \
  --sheets path/to/pinocchio.RUN.sheets.out \
  --plc-catalog path/to/pinocchio.RUN.plc.out \
  --mass-map-glob "path/to/pinocchio.RUN.massmap.seg*.fits" \
  --output-dir path/to/painted_nfw \
  --nfw-overdensity 200 \
  --nfw-reference-density critical \
  --mode derivatives \
  --mpi-plc-parts \
  --segment-workers 3
```

`submit.sh` defaults to `derivatives-profile` and accepts `SEGMENT_WORKERS`,
`GEPPETTO_MODE`, `OUTDIR`, `NFW_OVERDENSITY`, and
`NFW_REFERENCE_DENSITY` overrides. For a controlled Leonardo comparison,
submit separate one- and three-worker profile jobs on the same 30-rank,
three-CPU-per-rank allocation:

```bash
sbatch --export=ALL,SEGMENT_WORKERS=1,GEPPETTO_MODE=derivatives-profile,OUTDIR=/path/to/profile_w1 submit.sh
sbatch --export=ALL,SEGMENT_WORKERS=3,GEPPETTO_MODE=derivatives-profile,OUTDIR=/path/to/profile_w3 submit.sh
```

Profile jobs use one timing gather per segment to report
root-only stencil totals split into `query_disc`, compact lookup,
`pix2vec`/filter, concatenation, JAX transfer, and residual time. Phase timings
remain log-only. Adaptive branch, global-sample, retained-sample, and mass
conservation diagnostics are recorded in the manifest, never duplicated into
the lean NPZ arrays.

### Pipeline Modes

```text
--mode paint
    Paint and save the NFW particle-count map.

--mode derivatives
    Also save map-level derivatives wrt concentration amplitude, mass slope,
    and redshift slope.

--mode profile
    Paint and print timing information.

--mode derivatives-profile
    Compute derivatives and print timing information.
```

Calibration parameters exposed by the CLI:

```text
--concentration-amplitude
--concentration-mass-slope
--concentration-redshift-slope
--concentration-mass-pivot
--theta-resolution-rad
--n-resolution
```

The mass pivot is fixed when derivative maps are computed.

## Python API Overview

Core catalogue containers:

- `HaloCatalog`: comoving snapshot/box halo positions, masses, and redshifts.
- `LightconeHaloCatalog`: lightcone directions, comoving distances, masses, and
  redshifts.
- `LightconeSparseStencil`: fixed sparse halo-pixel pairs for PLC painting.
- `AdaptiveLightconeStencil`: complete global samples plus compact deposition
  rows for normalized particle-count assignment.

Core parameter containers:

- `ConcentrationParams`
- `NFWProfileParams`
- `TabulatedProjectedProfileParams`
- `Cosmology`

Main painters:

- `density_at_points`
- `paint_box_density_grid`
- `paint_lightcone_surface_density`
- `paint_lightcone_surface_density_sparse`
- `paint_lightcone_particle_count_map_sparse`
- `paint_lightcone_surface_density_tabulated_sparse`
- `paint_lightcone_particle_count_map_tabulated_sparse`

The default NFW concentration relation is a free power law,

```text
c(M, z) = A_c (M / M_pivot)^alpha_M (1 + z)^alpha_z
```

with convenience initializers `duffy08_all_200c()` and
`duffy08_relaxed_200c()`.

## Supported PINOCCHIO Inputs

PINOCCHIO readers live in `geppetto.io` and are intentionally outside the
differentiable JAX core.

Current reader support includes:

- snapshot halo catalogues, ASCII and native binary, including split files;
- full PLC catalogues, ASCII and native binary, using Cartesian positions for
  comoving distance and PINOCCHIO angular columns for map-basis directions;
- light PLC catalogues with angular coordinates and redshifts, ASCII and native
  32-byte binary;
- `HubbleTableFile` distance interpolation for light PLC conversion;
- parameter files, including particle-mass metadata;
- mass-sheet, `nz`, and mass-function ASCII outputs;
- compact HEALPix mass-map FITS tables.

Example:

```python
from geppetto.io import (
    healpix_pixel_area_sr,
    healpix_pixel_unit_vectors,
    read_pinocchio_hubble_table,
    read_pinocchio_lightcone_catalog,
    read_pinocchio_lightcone_light_catalog,
    read_pinocchio_parameter_file,
    read_pinocchio_snapshot_catalog,
)

metadata = read_pinocchio_parameter_file("parameter_file")

snapshot = read_pinocchio_snapshot_catalog("pinocchio.0.0000.example.catalog.out")
box_catalog = snapshot.to_halo_catalog(position="final")

plc = read_pinocchio_lightcone_catalog("pinocchio.example.plc.out")
lightcone_catalog = plc.to_lightcone_catalog(redshift="true")

distance = read_pinocchio_hubble_table("CAMBFiles/hubble.dat")
light_plc = read_pinocchio_lightcone_light_catalog("pinocchio.example.plc.out")
light_catalog = light_plc.to_lightcone_catalog(distance, redshift="true")

nside = int(metadata.parameters.get("MassMapNSIDE", ("256",))[0])
pixel_unit_vectors = healpix_pixel_unit_vectors(nside)
pixel_area = healpix_pixel_area_sr(nside)
```

Use `format="ascii"` or `format="binary"` to force a parser when auto-detection
is not desired.

For PLC angular directions, GEPPETTO follows PINOCCHIO's mass-map convention:
`theta` is latitude-like in degrees, `phi` is longitude, and the PLC axis is the
HEALPix north pole in the internal mass-map basis. This keeps halo catalogue
directions aligned with compact PINOCCHIO mass-map pixels.

## Differentiability Contract

The JAX core is differentiable with respect to profile parameters,
concentration parameters, halo masses, redshifts, distances, and continuous
positions used by the painters.

The following are fixed, non-differentiable geometry or I/O:

- file reading and parsing;
- halo selection by segment bounds;
- HEALPix indexing and `query_disc`;
- sparse sample rows, halo IDs, projected radii, and solid angles;
- adaptive branch selection, HEALPix query resolution, and hard support radius;
- mass-map compact pixel domains.

For map-level concentration derivatives, GEPPETTO uses a fixed sparse stencil
and forward-mode JVPs. It does not differentiate through the discrete halo-pixel
pair selection.

## Current Limitations

- Baryonification is a documented extension point, not implemented physics.
- Tabulated projected profiles are currently wired to sparse PLC painters, not
  dense PLC or 3D box painters.
- The tabulated profile parameterization is positive-only through
  `exp(log_shape)` and does not represent compensated signed profiles.
- The calibration script writes one output per PINOCCHIO mass-map segment and
  does not merge them into a global map.

More design context is in [docs/architecture.md](docs/architecture.md).

## Development and Validation

Run the standard checks from the repository root:

```bash
pytest
ruff check .
```

Validate the opt-in PINOCCHIO reader matrix against generated real outputs:

```bash
python scripts/validate_pinocchio_reader_matrix.py --all --overwrite
GEPPETTO_PINOCCHIO_READER_MATRIX_DIR=/tmp/geppetto-pinocchio-reader-validation pytest tests/test_pinocchio_reader_matrix.py
```

Useful runnable examples:

- `examples/nfw_box.py`
- `examples/nfw_lightcone.py`
- `examples/paint_halo_particles_for_pinocchio_segment.py`
- `examples/pinocchio_geppetto_case/README.md`

## Repository Layout

```text
GEPPETTO/
├── docs/architecture.md
├── examples/
├── scripts/validate_pinocchio_reader_matrix.py
├── src/geppetto/
│   ├── catalog.py
│   ├── concentration.py
│   ├── cosmology.py
│   ├── geometry.py
│   ├── io.py
│   ├── painters.py
│   └── profiles.py
└── tests/
```

## Near-Term Priorities

1. Validate adaptive NFW assignment against high-resolution reference maps.
2. Expand validation coverage for PINOCCHIO reader and mass-map workflows.
3. Add scalable, reusable HEALPix-local sparse stencil builders outside the JAX
   core.
4. Generalize sparse painting toward compensated and baryonified profile
   families.
5. Define the production convention for combining GEPPETTO one-halo maps with
   PINOCCHIO two-halo maps.

## License

GEPPETTO is distributed under the MIT license. See [LICENSE](LICENSE).
