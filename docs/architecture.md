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

For PINOCCHIO mass-map integration, the HEALPix-facing painter returns a
count-equivalent one-halo collector:

```text
count_map = Sigma_NFW(R_perp) * chi_h^2 * Omega_pix / m_particle
```

where `m_particle` is computed from the PINOCCHIO parameter file as the mean
comoving matter density times `BoxSize^3 / GridSize^3`, with `BoxInH100`
controlling whether the box size is already in `Mpc/h`. HEALPix pixel-to-vector
conversion remains in `geppetto.io`; the JAX painter only sees fixed unit
vectors and a pixel area.

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
  PLCs, including split files. Light PLC readers support angular/redshift ASCII
  and 32-byte binary outputs, but conversion to GEPPETTO lightcone catalogues
  requires an explicit distance interpolator from the PINOCCHIO
  `HubbleTableFile`.
- HEALPix helpers are I/O adapters only; HEALPix indices are not differentiable
  targets and do not enter JAX kernels.
- No exact mass-conserving smooth truncation yet.
- Projected NFW uses a tapered untruncated analytic projected profile rather than the full truncated projected NFW expression.
- Baryonification is a documented extension point, not implemented physics.
