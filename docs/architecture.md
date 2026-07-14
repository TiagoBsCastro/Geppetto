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

The three-dimensional profile has hard support at `R_delta` and is normalized
to the input halo mass inside that sphere. The projected NFW kernel is the
exact finite line-of-sight integral through the same hard-truncated sphere,
not an infinite projected profile with a two-dimensional cutoff. A tiny
explicit central-radius softening regularizes the NFW cusp; the support test uses
the unsoftened radius and is independent of concentration.

## PLC mode

The initial PLC painter evaluates an approximate projected NFW surface density on supplied pixel unit vectors. For each halo-pixel pair it computes a transverse comoving separation using the chord approximation:

```text
R_perp = chi_h sqrt[2(1 - nhat_pixel . nhat_halo)]
```

This is appropriate for compact one-halo profiles and avoids `arccos` near zero angle. Exact angular-distance modes can be added if required.

The sparse PLC painter uses the same NFW projected profile, but receives a
precomputed halo-pixel stencil:

```text
(pix_id, halo_id, R_perp, optional pair_weight)
```

Stencil construction is non-core geometry. It may use NumPy, HEALPix helpers, or
survey masks outside JAX, then pass fixed pair arrays into the differentiable
painter. `LightconeSparseStencil` stores `n_pix` as static pytree metadata so
the sparse painter can allocate its `(n_pix,)` result under `jax.jit`.
`pair_weight` is dimensionless and defaults to one; production JIT buckets use
zero weights to make padded duplicate indices inert. The first builder,
`build_lightcone_sparse_stencil_bruteforce`, retains pairs with
`R_perp <= Rmax_halo`; `Rmax` and the retained pair set are not differentiable
parameters. This helper materializes the full `n_pix * n_halo` separation
matrix and is intended for validation and small maps. A scalable HEALPix-local
builder should return the same stencil container after doing the discrete pixel
queries outside JAX. The sparse painter remains differentiable with respect to
concentration and profile parameters because it only gathers halo fields,
evaluates the projected profile, and scatter-adds into the output map.

The production PINOCCHIO segment workflow uses
`AdaptiveLightconeStencil`. It retains every global profile sample needed for
normalization and records compact deposition separately. Selected-catalogue
halo indices are remapped to the constant rank-local catalogue, arrays are
padded to power-of-two buckets, and profile samples are evaluated in static
chunks. In concentration-derivative mode one linearization returns the primal
map and applies the three JVP directions without repeating the primal paint.

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

For PINOCCHIO mass-map integration, each resolved global sample has raw weight:

```text
q_ha = Sigma_NFW,truncated(R_perp) * chi_h^2 * Omega_sample
```

JAX sums `q_ha` over complete global support per halo, normalizes by that sum,
and multiplies by `M_h / m_particle`. Compact filtering happens only after
normalization. The complete assignment conserves catalogue mass exactly at map
level; compact output can retain less at an angular footprint boundary.
`m_particle` is read from PINOCCHIO metadata. HEALPix indexing remains outside
the differentiable core.

PINOCCHIO mass-map pixels are expressed in the internal PLC angular basis, with
the PLC axis at the HEALPix north pole. GEPPETTO therefore converts PINOCCHIO
PLC `theta, phi` columns directly to map-basis unit vectors for PLC painting.
The full PLC reader still uses Cartesian positions to compute radial distance
`chi`, but not to define angular map directions.

The support angle is the chord-consistent expression
`2 asin(min(1, R_delta / (2 chi)))`. Unresolved halos use NGP. Intermediate
halos query child centers directly at the derived NESTED refinement level;
well-resolved halos query native RING centers. Both query branches use
`inclusive=False`, expand the angular radius by one floating-point step, and
apply the exact chord-distance cut `R_perp <= R_delta`. Refined children are
aggregated through their native NESTED parent and then converted to RING.

The same calibration script has an opt-in mixed parallel mode. In
`--mpi-plc-parts` mode, rank `r` reads only `--plc-catalog.r`; the number of MPI
ranks must match the number of contiguous split PLC parts. Each rank computes
partial compact segment maps, then sums additive arrays and diagnostics for each
segment on rank 0 before writing the final compressed NPZ output. At
`--segment-workers 1`, MPI mode is fully streamed and holds one segment payload
at a time. At higher worker counts, it uses a bounded ordered pipeline: each
rank computes up to `N` segments ahead, but reductions and writes remain
serialized by segment index. Segment-level parallelism inside each rank uses
`--segment-workers N`, a shared-memory thread pool over mass-map segments. The
workers form a bounded prefetch window: they can overlap later segment work
with MPI waiting, but they do not directly parallelize the Python per-halo
`healpy.query_disc` loop inside one segment. Retained map memory scales roughly
linearly with the worker count. In profile modes, one small timing buffer per
segment is gathered to rank 0 to report rank compute, result-wait, reduction,
and stencil-phase min/mean/max values. Stencil phases cover `query_disc`,
compact lookup, `pix2vec`/filter, concatenation, JAX transfer, and residual host
work. Profile-only phase values remain log-only; normal paint and derivative
modes add no timing collective.

The NPZ payload contains only the painted NFW count-equivalent array and, when
requested, its three concentration-parameter derivative arrays. Compact pixel
IDs, HEALPix metadata, and source map values remain in the original PINOCCHIO
FITS segment. The CSV manifest records the source segment path and scientific
parameters needed to interpret the row-aligned arrays. It also records selected
halo count and mass, expected and assigned global counts, retained and
outside-compact counts, adaptive branch and profile-sample counts,
supersampling levels, MPI rank count, segment-worker count, and Git commit.
Timing diagnostics are omitted.

The NFW spherical-overdensity mass definition is explicit at the command line.
Users select a finite positive constant `Delta`, the redshift-dependent
Bryan--Norman virial threshold, or a one-dimensional `.npy` array with one
`Delta` per sheet row. Each mode also selects critical or mean reference
density. Bryan--Norman mean overdensities are divided by `Omega_m(z)`, so
the critical- and mean-reference choices represent the same physical threshold.
PINOCCHIO catalogue masses are interpreted in the selected definition without
conversion, and the manifest records that interpretation.

Discovered all-segment inputs must be one contiguous range within the sheet
table. Segment selection remains half-open unless the segment is the actual
physical final sheet, `len(sheets) - 1`; the highest member of a partial glob
is not made inclusive. The explicit inclusive-upper override is confined to
single-segment execution.
Radial membership is determined by the halo centre. A selected halo's complete
projected profile belongs to that segment and is not clipped or divided at
radial sheet boundaries.

MPI computation errors include rank, segment, and source-map context before
calling `Comm.Abort`. This prevents ranks that reached a map reduction from
waiting indefinitely when another rank fails during local I/O, stencil
construction, JAX transfer, or painting.

## Production calibration review closure

The actionable engineering findings from the 2026 internal review of the
PINOCCHIO calibration workflow are closed. This records implementation status,
not the removal of the physical-model limitations listed below.

1. **Documentation and resource requests.** The user-facing documentation and
   Leonardo submission example distinguish bounded segment prefetch from
   parallel execution of the Python per-halo `healpy.query_disc` loop and
   document the memory cost of additional workers. Static profile-sample
   chunking is identified as a numerical memory control.
2. **Precision policy.** The reusable GEPPETTO core follows the caller's JAX
   precision configuration. The production PINOCCHIO segment script configures
   float64 before importing the JAX-heavy modules and offers explicit
   `--jax-precision float32` opt-in for memory-constrained runs. The selected
   precision is preserved by painting and MPI reduction and recorded in the
   manifest.
3. **Order-safe compact-pixel indexing.** Each segment builds one
   `MassMapPixelIndex` and reuses it for local halo binning and sparse-stencil
   construction. The index uses a bounded dense inverse map when appropriate
   and a sorted `searchsorted` fallback otherwise. Both backends preserve the
   arbitrary row order of the compact source map and return `-1` for pixels not
   present in the segment.
4. **MPI reduction efficiency.** Full map and derivative arrays use buffer
   `Comm.Reduce` collectives. Additive integer and floating-point diagnostics
   are packed into numeric reductions, derived count ratios are recomputed from
   reduced values, and only rank 0 allocates receive buffers or writes output.
   No object reduction or temporary per-rank map file remains in the production
   path.
5. **Segment workers and load balance.** Profiling supports the retained
   bounded, ordered thread pipeline. Workers prefetch complete segments while
   the main thread reduces and writes in segment order; one worker provides the
   minimum-memory streaming mode. A process pool was not introduced because it
   would duplicate rank-local catalogues and maps without addressing the
   Python/GIL-bound stencil query directly.
6. **Sparse JIT bucketing.** Production adaptive stencils are remapped to the
   constant rank-local catalogue and padded to power-of-two sample buckets with
   invalid zero-area samples. Module-level JIT kernels are reused, and
   derivative mode obtains the primal map and three concentration JVPs from one
   linearization. Tests cover both supported floating-point precisions, inert
   padding, empty stencils, bucket reuse, and global mass conservation.

The review's correctness checks that did not require code changes were also
closed explicitly. The chord-distance query radius is the algebraic inverse of
the exact retained-pair filter; NumPy and JAX scatter-adds correctly accumulate
duplicate indices; zero-weight padding is inert; and segment and split-PLC
discovery validate contiguous indices and MPI rank-count matching.

The associated storage cleanup is also complete. The obsolete selectable MPI
output mode and rank-local output files were removed; MPI sum-to-root is the
only distributed output path. The workflow writes compressed NPZ arrays only,
does not duplicate source FITS pixels or metadata, and does not store
derivative-of-map-sum diagnostics. The compact CSV manifest retains only the
provenance, parameters, and compact scientific diagnostics needed to interpret
and audit the row-aligned arrays.

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
- Resolved PLC deposition samples profiles at HEALPix pixel centers. Local
  supersampling reduces discretization error for intermediate angular sizes,
  but convergence still depends on `n_resolution` and should be validated for
  each production NSIDE.
- Tabulated projected profiles are currently wired only to sparse PLC painters.
- Baryonification is a documented extension point, not implemented physics.
