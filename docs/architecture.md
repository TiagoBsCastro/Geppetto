# GEPPETTO architecture notes

## Conceptual boundary

GEPPETTO is not a replacement for PINOCCHIO lightcones. It is the differentiable one-halo layer placed on top of PINOCCHIO halo catalogues and maps.

PINOCCHIO side:

- halo catalogues;
- PLC geometry;
- large-scale/two-halo HEALPix density maps;
- survey-volume backbone.

GEPPETTO side:

- one-halo analytic density profiles;
- differentiable profile parameters;
- box-grid, lightcone-pixel, and one-halo HEALPix map-value evaluation;
- future baryonic profile transforms.

## Core abstraction

The central primitive is:

```python
density_at_points(points, catalog, ...)
```

Everything else is geometry:

- a box grid is just a set of Cartesian target points;
- a lightcone angular map is a set of target pixel unit vectors plus a projected profile prescription;
- a shell map can later be represented by pixel vectors, shell distances, and shell widths.

This keeps the profile code independent of HEALPix, file formats, and survey masks.

## Why pixel vectors rather than HEALPix indices?

HEALPix index arithmetic is discrete and not a useful differentiation target. GEPPETTO therefore accepts fixed pixel unit vectors. Gradients are meaningful with respect to halo/profile/cosmological parameters, while the map geometry is treated as a fixed evaluation grid.

## Current NFW model

The initial NFW model is:

```text
rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2]
```

with:

```text
r_s = R_delta / c(M,z)
c(M,z) = A (M / M_pivot)^B (1 + z)^C
```

The profile is normalized to integrate to the input halo mass inside `R_delta` before the optional smooth taper. The smooth taper is enabled by default because it is better behaved for gradient-based workflows than a hard step at `R_delta`.

## PLC mode

The initial PLC painter evaluates an approximate projected NFW surface density on supplied pixel unit vectors. For each halo-pixel pair it computes a transverse comoving separation using the chord approximation:

```text
R_perp = chi_h sqrt[2(1 - nhat_pixel . nhat_halo)]
```

This is appropriate for compact one-halo profiles and avoids `arccos` near zero angle. Exact angular-distance modes can be added if required.

The sparse PLC painter uses the same NFW projected profile, but receives a
precomputed halo-pixel stencil:

```text
(pix_id, halo_id, R_perp)
```

Stencil construction is non-core geometry. It may use NumPy, HEALPix helpers, or
survey masks outside JAX, then pass fixed pair arrays into the differentiable
painter. `LightconeSparseStencil` stores `n_pix` as static pytree metadata so
the sparse painter can allocate its `(n_pix,)` result under `jax.jit`. The first
builder, `build_lightcone_sparse_stencil_bruteforce`, retains pairs with
`R_perp <= Rmax_halo`; `Rmax` and the retained pair set are not differentiable
parameters. This helper materializes the full `n_pix * n_halo` separation
matrix and is intended for validation and small maps. A scalable HEALPix-local
builder should return the same stencil container after doing the discrete pixel
queries outside JAX. The sparse painter remains differentiable with respect to
concentration and profile parameters because it only gathers halo fields,
evaluates the projected profile, and scatter-adds into the output map.

The first non-NFW profile path is sparse-PLC only:

```text
Sigma(R) = M_halo * shape(R / Rmax) /
           [Rmax^2 * 2 pi integral x shape(x) dx]
```

`TabulatedProjectedProfileParams` stores a shared dimensionless radius grid and
unconstrained `log_shape` values. The JAX kernel exponentiates `log_shape`,
linearly interpolates the positive template, sets values outside the fixed
support to zero, and normalizes the projected mass inside `Rmax` to the halo
mass. `x` and `Rmax` are fixed geometry and are stopped from participating in
gradients. Call `validate_tabulated_projected_profile_params` outside JAX paths
for manually constructed tabulated profiles. Dense PLC and box-grid painters
remain NFW-only in this first tabulated-profile release.

This normalization is continuous: exact discrete mass conservation after
HEALPix/pixel-center sampling is not enforced. The current `exp(log_shape)`
parameterization represents positive projected profiles only; compensated signed
profiles require a later parameterization.

For PINOCCHIO mass-map integration, the HEALPix-facing painter returns a
count-equivalent one-halo collector:

```text
count_map = Sigma_NFW(R_perp) * chi_h^2 * Omega_pix / m_particle
```

where `m_particle` is computed from the PINOCCHIO parameter file as the mean
comoving matter density times `BoxSize^3 / GridSize^3`, with `BoxInH100`
controlling whether the box size is already in `Mpc/h`. HEALPix pixel-to-vector
conversion remains in `geppetto.io`; the JAX painter only sees fixed unit
vectors and a pixel area. The sparse equivalent,
`paint_lightcone_particle_count_map_sparse`, uses the same mass-per-pixel divided
by particle-mass convention on a precomputed stencil. The tabulated sparse
equivalent follows the same count convention with the tabulated projected
profile replacing `Sigma_NFW`.

PINOCCHIO mass-map pixels are expressed in the internal PLC angular basis, with
the PLC axis at the HEALPix north pole. GEPPETTO therefore converts PINOCCHIO
PLC `theta, phi` columns directly to map-basis unit vectors for PLC painting.
The full PLC reader still uses Cartesian positions to compute radial distance
`chi`, but not to define angular map directions.

The PINOCCHIO calibration script also has hidden sparse-stencil audit flags for
benchmarking HEALPix query choices without changing the default scientific
path:

```bash
python examples/paint_halo_particles_for_pinocchio_segment.py ... \
  --mode paint \
  --stencil-diagnostics \
  --stencil-query-mode inclusive

python examples/paint_halo_particles_for_pinocchio_segment.py ... \
  --mode paint \
  --stencil-diagnostics \
  --stencil-query-mode center

python examples/paint_halo_particles_for_pinocchio_segment.py ... \
  --mode paint \
  --stencil-compare-query-modes
```

`inclusive` keeps the current `healpy.query_disc(..., inclusive=True)` path.
`center` uses `inclusive=False`. The comparison mode is single-segment only and
reports stencil timing, query counts, kept-pair counts, and painted-map
differences.

The same calibration script has an opt-in mixed parallel mode. In
`--mpi-plc-parts` mode, rank `r` reads only `--plc-catalog.r`; the number of MPI
ranks must match the number of contiguous split PLC parts. Each rank computes
partial compact segment maps, then sums additive arrays and diagnostics for each
segment on rank 0 before writing the final NPZ/FITS outputs. At
`--segment-workers 1`, MPI mode is fully streamed and holds one segment payload
at a time. At higher worker counts, it uses a bounded ordered pipeline: each
rank computes up to `N` segments ahead, but reductions and writes remain
serialized by segment index. Segment-level parallelism inside each rank uses
`--segment-workers N`, a shared-memory thread pool over mass-map segments. The
workers form a bounded prefetch window: they can overlap later segment work
with MPI waiting, but they do not directly parallelize the Python per-halo
`healpy.query_disc` loop inside one segment. Retained map memory scales roughly
linearly with the worker count. In profile modes, one small timing buffer per
segment is gathered to rank 0 to report rank compute, result-wait, and reduction
min/mean/max values; normal paint and derivative modes add no timing
collective.

## Box mode

The box painter constructs cell-centre positions and evaluates the 3D profile with optional periodic minimum-image wrapping. This is intended for snapshot-box validation and for measuring the matter power spectrum from the painted one-halo density field.

## Baryonification plan

Baryonification should be implemented as a profile family with the same interface as the NFW functions:

```python
rho = baryonified_density(r, mass, redshift, cosmology, baryon_params, ...)
sigma = baryonified_projected_surface_density(r_perp, mass, redshift, cosmology, baryon_params, ...)
```

The painter should dispatch on the profile function or receive a callable profile kernel. This avoids duplicating geometry code.

## Known limitations in version 0.1.0

- PINOCCHIO catalogue readers support ASCII and native binary catalogues/full
  PLCs, including split files. Full PLC angular directions follow the
  PINOCCHIO mass-map basis, where `theta` is latitude-like and `phi` is
  longitude. Light PLC readers support angular/redshift ASCII and 32-byte
  binary outputs, but conversion to GEPPETTO lightcone catalogues requires an
  explicit distance interpolator from the PINOCCHIO `HubbleTableFile`.
- HEALPix helpers are I/O adapters only; HEALPix indices are not differentiable
  targets and do not enter JAX kernels.
- No exact mass-conserving smooth truncation yet.
- Projected NFW uses a tapered untruncated analytic projected profile rather than the full truncated projected NFW expression.
- Tabulated projected profiles are currently wired only to sparse PLC painters.
- Baryonification is a documented extension point, not implemented physics.
