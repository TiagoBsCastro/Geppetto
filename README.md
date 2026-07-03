# GEPPETTO

**GEPPETTO** is a fully differentiable JAX framework for painting one-halo matter contributions around PINOCCHIO halo catalogues.

PINOCCHIO remains the halo/lightcone backbone and can already provide the large-scale/two-halo HEALPix density maps. GEPPETTO is the complementary one-halo painter: it adds analytic, differentiable halo-profile contributions on top of catalogues from either past-light-cone outputs or comoving snapshot boxes.

Current status: **pre-alpha repository starter**.

## First implemented model

The first profile prescription is NFW with a Duffy-like power-law concentration--mass relation,

```text
c(M, z) = A (M / M_pivot)^B (1 + z)^C
```

where `A`, `B`, `C`, and `M_pivot` are ordinary JAX parameters and can be varied, differentiated, calibrated, or sampled. The package includes two convenience initializers:

- `duffy08_all_200c()`
- `duffy08_relaxed_200c()`

The default physical convention is `M_200c`; masses are in `Msun/h`, positions and distances are in comoving `Mpc/h`.

## Long-term goal

The long-run target is to support an Aricò-style baryonification prescription through differentiable profile transforms for dark matter, gas, stars, and ejected gas. This repository intentionally starts with a clean NFW implementation and a documented baryonification extension point rather than a partial baryonic model.

## Installation

GEPPETTO is currently installed from the GitHub repository. The recommended
development setup uses Miniforge/Mambaforge plus an editable pip install:

```bash
git clone git@github.com:TiagoBsCastro/Geppetto.git
cd Geppetto

mamba create -n geppetto-dev python=3.12
mamba activate geppetto-dev

python -m pip install -e '.[io,dev]'
```

Use the extras according to the workflow:

- `python -m pip install -e .` installs the differentiable JAX/NumPy core.
- `python -m pip install -e '.[io]'` also installs FITS, HEALPix, HDF5, and
  PINOCCHIO reader dependencies: `astropy`, `healpy`, and `h5py`.
- `python -m pip install -e '.[dev]'` installs test, lint, and type-check tools.
- `python -m pip install -e '.[io,dev]'` is recommended for developing and
  running the PINOCCHIO examples.

Verify the installation:

```bash
python -c "import geppetto; print(geppetto.__version__)"
pytest
ruff check .
```

For GPU use, install the JAX build appropriate for your CUDA stack before
installing GEPPETTO. Follow the official JAX installation selector rather than
pinning a CUDA command from this README.

## Minimal comoving-box example

```python
import jax.numpy as jnp
from geppetto import HaloCatalog, Cosmology, duffy08_all_200c, paint_box_density_grid

catalog = HaloCatalog(
    position=jnp.array([[50.0, 50.0, 50.0]]),  # Mpc/h
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

## Minimal PLC/lightcone example

GEPPETTO does not require `healpy` inside the differentiable core. Pass HEALPix pixel unit vectors from PINOCCHIO or another map layer.

```python
import jax.numpy as jnp
from geppetto import LightconeHaloCatalog, paint_lightcone_surface_density

pixel_unit_vectors = jnp.array([
    [1.0, 0.0, 0.0],
    [0.999, 0.045, 0.0],
])

catalog = LightconeHaloCatalog(
    unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
    chi=jnp.array([1000.0]),      # Mpc/h
    mass=jnp.array([1.0e14]),     # Msun/h
    redshift=jnp.array([0.3]),
)

sigma = paint_lightcone_surface_density(pixel_unit_vectors, catalog)
print(sigma)  # projected one-halo surface density at the supplied pixels
```

For larger maps, build a fixed sparse halo-pixel stencil outside the
differentiable core and paint only the retained pairs:

```python
from geppetto import paint_lightcone_surface_density_sparse
from geppetto.io import build_lightcone_sparse_stencil_bruteforce

stencil = build_lightcone_sparse_stencil_bruteforce(
    pixel_unit_vectors,
    catalog,
    rmax_mpc_h=5.0,  # fixed geometry cut in comoving Mpc/h
)
sigma_sparse = paint_lightcone_surface_density_sparse(stencil, catalog)
```

The sparse painter is differentiable with respect to NFW concentration/profile
parameters. Pixel indices, HEALPix geometry, and the `Rmax` stencil cut are fixed
inputs and are not differentiation targets.

`build_lightcone_sparse_stencil_bruteforce` materializes an `n_pix * n_halo`
separation matrix before filtering, so it is meant for validation, examples, and
small maps. Production HEALPix-local stencil construction belongs outside the
JAX painter and should pass the same `LightconeSparseStencil` container.

## Tabulated sparse lightcone profiles

For sparse PLC maps, GEPPETTO also supports a positive mass-normalized projected
profile template. The template is tabulated in dimensionless radius
`x = R / Rmax`; `Rmax` is fixed geometry and is not differentiated.

```python
import jax.numpy as jnp

from geppetto import (
    TabulatedProjectedProfileParams,
    paint_lightcone_surface_density_tabulated_sparse,
)
from geppetto.io import validate_tabulated_projected_profile_params

profile = TabulatedProjectedProfileParams(
    x=jnp.linspace(0.0, 1.0, 32),
    log_shape=jnp.zeros(32),  # differentiable unconstrained amplitudes
)
validate_tabulated_projected_profile_params(profile)

sigma_tabulated = paint_lightcone_surface_density_tabulated_sparse(
    stencil,
    catalog,
    rmax_mpc_h=5.0,  # scalar or one value per halo
    profile_params=profile,
)
```

The tabulated sparse painter uses JAX interpolation, exponentiates `log_shape`
for positivity, and normalizes the disk integral inside `Rmax` to the halo mass.
It treats both `x` and `Rmax` as fixed geometry and is differentiable with
respect to `log_shape`, masses, and catalogue distances used for mass-per-pixel
conversion.

The profile is normalized in the continuum. Exact discrete mass conservation on
a HEALPix/pixel stencil is not enforced in this release, so coarse maps or
stencils with very small `Rmax` only approximate the halo mass after pixel
sampling. The current `exp(log_shape)` parameterization also enforces positive
profiles; compensated signed profiles are a later extension.

## HEALPix one-halo particle-count map

For PINOCCHIO mass-map integration, GEPPETTO can paint an NFW one-halo mass
collector in particle-count-equivalent units. The core still receives fixed
pixel vectors; `geppetto.io` supplies the optional `healpy` adapter and parses
the PINOCCHIO parameter file to compute the grid-element particle mass.

```python
from geppetto import (
    paint_lightcone_particle_count_map,
    paint_lightcone_particle_count_map_sparse,
    paint_lightcone_particle_count_map_tabulated_sparse,
)
from geppetto.io import (
    healpix_pixel_area_sr,
    healpix_pixel_unit_vectors,
    read_pinocchio_parameter_file,
)

metadata = read_pinocchio_parameter_file("parameter_file")
pixel_unit_vectors = healpix_pixel_unit_vectors(nside=256, nest=False)

one_halo_counts = paint_lightcone_particle_count_map(
    pixel_unit_vectors,
    lightcone_catalog,
    particle_mass_msun_h=metadata.particle_mass_msun_h,
    pixel_area_sr=healpix_pixel_area_sr(256),
    cosmology=metadata.cosmology,
    chunk_size=1024,
)

sparse_counts = paint_lightcone_particle_count_map_sparse(
    stencil,
    lightcone_catalog,
    particle_mass_msun_h=metadata.particle_mass_msun_h,
    pixel_area_sr=healpix_pixel_area_sr(256),
    cosmology=metadata.cosmology,
)

tabulated_sparse_counts = paint_lightcone_particle_count_map_tabulated_sparse(
    stencil,
    lightcone_catalog,
    rmax_mpc_h=rmax_per_halo,
    profile_params=profile,
    particle_mass_msun_h=metadata.particle_mass_msun_h,
    pixel_area_sr=healpix_pixel_area_sr(256),
)
```

`one_halo_counts`, `sparse_counts`, and `tabulated_sparse_counts` are one-halo
mass per pixel divided by the PINOCCHIO particle mass. They do not include
PINOCCHIO's two-halo map.

## PINOCCHIO c-M calibration script

`examples/paint_halo_particles_for_pinocchio_segment.py` runs a compact
PINOCCHIO-to-NFW painting pipeline for calibrating a concentration--mass
relation. It can run on a single PINOCCHIO mass-map segment or on all existing
segments discovered from a glob. For each segment, it bins the resolved halo
mass as a point-halo diagnostic, paints an NFW one-halo
particle-count-equivalent map on the same compact HEALPix pixel domain, and can
save map-level derivatives with respect to concentration amplitude, mass slope,
and redshift slope. Segment-local compatibility is preserved: the compact
`PIXEL` list, row ordering, `NSIDE`, `ORDERING`, and segment bounds match the
corresponding PINOCCHIO on-the-fly mass-map segment.

The script always saves `nfw_particle_counts` and uses one public execution
mode flag:

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params examples/pinocchio_geppetto_case/parameter_file \
  --sheets examples/pinocchio_geppetto_case/pinocchio.example.sheets.out \
  --mass-map examples/pinocchio_geppetto_case/pinocchio.example.massmap.seg001.fits \
  --plc-catalog examples/pinocchio_geppetto_case/pinocchio.example.plc.out \
  --sheet-index 1 \
  --output examples/pinocchio_geppetto_case/halo_particles.seg001.paint.npz
```

Use derivative mode to also save the three derivative maps needed for a c-M
calibration loop:

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params examples/pinocchio_geppetto_case/parameter_file \
  --sheets examples/pinocchio_geppetto_case/pinocchio.example.sheets.out \
  --mass-map examples/pinocchio_geppetto_case/pinocchio.example.massmap.seg001.fits \
  --plc-catalog examples/pinocchio_geppetto_case/pinocchio.example.plc.out \
  --sheet-index 1 \
  --output examples/pinocchio_geppetto_case/halo_particles.seg001.derivs.npz \
  --mode derivatives
```

To paint all discovered mass-map segments, provide a glob and an output
directory. This writes one `painted_nfw.segXXX.npz`, one compact
`painted_nfw.segXXX.fits`, and a `painted_nfw_manifest.csv` summary:

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py \
  --params examples/pinocchio_geppetto_case/parameter_file \
  --sheets examples/pinocchio_geppetto_case/pinocchio.example.sheets.out \
  --plc-catalog examples/pinocchio_geppetto_case/pinocchio.example.plc.out \
  --mass-map-glob "examples/pinocchio_geppetto_case/pinocchio.example.massmap.seg*.fits" \
  --output-dir examples/pinocchio_geppetto_case/painted_nfw \
  --mode derivatives
```

For timing output, use `--mode profile` or `--mode derivatives-profile`.
Concentration calibration parameters are exposed as
`--concentration-amplitude`, `--concentration-mass-slope`,
`--concentration-redshift-slope`, and `--concentration-mass-pivot`; the pivot is
fixed when computing derivative maps.

## Reading PINOCCHIO outputs

PINOCCHIO readers live in `geppetto.io`, outside the differentiable core. They
preserve raw metadata and provide explicit conversions to GEPPETTO catalogues.
Snapshot and PLC readers auto-detect ASCII output and native PINOCCHIO binary
output, including split binary files such as `*.catalog.out.0`:

```python
from geppetto.io import (
    healpix_pixel_area_sr,
    healpix_pixel_unit_vectors,
    read_pinocchio_parameter_file,
    read_pinocchio_hubble_table,
    read_pinocchio_lightcone_catalog,
    read_pinocchio_lightcone_light_catalog,
    read_pinocchio_snapshot_catalog,
)

snapshot = read_pinocchio_snapshot_catalog("pinocchio.0.0000.example.catalog.out")
box_catalog = snapshot.to_halo_catalog(position="final")

plc = read_pinocchio_lightcone_catalog("pinocchio.example.plc.out")
lightcone_catalog = plc.to_lightcone_catalog(redshift="true")

distance = read_pinocchio_hubble_table("CAMBFiles/hubble.dat")
light_plc = read_pinocchio_lightcone_light_catalog("pinocchio.example.plc.out")
light_lightcone_catalog = light_plc.to_lightcone_catalog(distance, redshift="true")

metadata = read_pinocchio_parameter_file("parameter_file")
nside = int(metadata.parameters.get("MassMapNSIDE", ("256",))[0])
pixel_unit_vectors = healpix_pixel_unit_vectors(nside)
pixel_area = healpix_pixel_area_sr(nside)
```

Use `format="ascii"` or `format="binary"` to force a specific parser. The
full PLC reader expects Cartesian positions and can convert directly. Light PLC
readers support ASCII and native 32-byte binary output; because those files only
store angles and redshifts, conversion requires an explicit
`read_pinocchio_hubble_table` distance interpolator. The interpolator reads the
PINOCCHIO `HubbleTableFile` convention, `E(z) = H(z) / H0`, and returns
comoving distances in `Mpc/h`.

Mass-map FITS readers and HEALPix helpers are also in `geppetto.io` and import
Astropy/healpy lazily, so install with `.[io]` when reading FITS products or
generating HEALPix pixel vectors.

To validate the readers against real PINOCCHIO outputs, run the opt-in matrix
driver. It creates ASCII/binary and single/split output cases in `/tmp`:

```bash
python scripts/validate_pinocchio_reader_matrix.py --all --overwrite
GEPPETTO_PINOCCHIO_READER_MATRIX_DIR=/tmp/geppetto-pinocchio-reader-validation pytest tests/test_pinocchio_reader_matrix.py
```

## Differentiability contract

The core kernels are written in JAX and are differentiable with respect to profile parameters, concentration parameters, masses, redshifts, and continuous coordinates. Pixel indices and catalogue selection are discrete by construction, so they are kept outside the differentiable core.

The current NFW implementation uses a smooth radial taper by default to avoid a hard non-differentiable profile edge. Exact mass-conserving truncation and baryonified compensated profiles are future model options.

## Repository layout

```text
GEPPETTO/
├── AGENTS.md                 # instructions for future coding agents
├── docs/architecture.md       # design notes and roadmap
├── examples/                  # runnable examples
├── src/geppetto/              # package source
│   ├── baryonification.py     # extension point for Aricò-style model
│   ├── catalog.py             # JAX pytree catalogue containers
│   ├── concentration.py       # Duffy-like c-M relations
│   ├── cosmology.py           # background densities
│   ├── geometry.py            # box/lightcone geometry
│   ├── io.py                  # non-core catalogue adapters
│   ├── painters.py            # one-halo painting kernels
│   └── profiles.py            # NFW 3D and projected profiles
└── tests/                     # differentiability and shape tests
```

## Development priorities

1. Validate NFW mass normalization and projection conventions against controlled analytic cases.
2. Add robust PINOCCHIO ASCII/FITS/HDF5 readers in `geppetto.io` without polluting the differentiable core.
3. Add high-throughput chunking over pixels and haloes for large PLC maps.
4. Implement compensated/baryonified profiles following the selected Aricò et al. prescription.
5. Add direct integration hooks to your PINOCCHIO fork once the catalogue/map boundary is fixed.
