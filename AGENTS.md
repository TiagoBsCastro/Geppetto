# Agent instructions for GEPPETTO

## Mission

GEPPETTO is a differentiable JAX one-halo matter/profile painter for PINOCCHIO halo catalogues. PINOCCHIO supplies the large-scale halo/lightcone backbone and can already produce the two-halo HEALPix density maps. GEPPETTO must add local one-halo profile contributions around catalogued haloes and must work for both:

1. past-light-cone catalogues and angular maps; and
2. comoving snapshot boxes and 3D grids.

The first supported physical model is NFW with a Duffy-like analytical concentration--mass relation whose parameters are free. The long-term target is an Aricò-style baryonification module.

## Non-negotiable coding rules

1. **JAX-first core**: anything in `profiles.py`, `geometry.py`, `cosmology.py`, `concentration.py`, and `painters.py` must use `jax.numpy`, `jax.lax`, `jax.vmap`, or other JAX-compatible primitives. Do not use ordinary NumPy inside differentiable kernels.
2. **No hidden I/O in kernels**: file readers, HEALPix indexing, FITS/HDF5 parsing, unit conversion from external formats, and plotting belong outside the differentiable core.
3. **Keep catalogue containers as pytrees**: use `NamedTuple`, frozen dataclasses registered as pytrees, or simple dictionaries of JAX arrays. Avoid mutable classes in core APIs.
4. **Profile parameters must be explicit**: all physical parameters must appear in `NamedTuple`/pytree parameter containers. Do not hard-code calibration constants except as named defaults or factory functions.
5. **Do not mix the two-halo term into GEPPETTO**: the two-halo density map is handled by the PINOCCHIO fork. GEPPETTO should return one-halo contributions that can be added to the existing map or grid layer.
6. **Differentiability tests are mandatory**: every new model must have at least one `jax.grad` or `jax.jacfwd` test showing finite gradients with respect to its parameters.
7. **Shape tests are mandatory**: every public painter must have a test for expected output shapes on small synthetic catalogues.
8. **Document units**: public functions must state whether inputs are comoving `Mpc/h`, physical `Mpc/h`, `Msun/h`, surface density, volume density, or dimensionless density contrast.
9. **Avoid host callbacks**: no `print`, `np.asarray`, `list(array)`, file writes, or Python side effects inside JIT-compiled paths.
10. **Large maps need chunking**: new PLC or box painters must expose static chunk sizes over halo and/or pixel dimensions to avoid full `N_pix × N_halo` memory explosions.

## Physical conventions in the current release

- Mass unit: `Msun/h`.
- Coordinate unit: comoving `Mpc/h`.
- Default halo mass definition: `M_200c`.
- Default concentration relation: Duffy-like 200c full-sample relation, implemented as a free power law.
- NFW radii are converted to comoving radii before profile evaluation.
- NFW density has hard support at `R_delta`; projected NFW is the exact finite
  line-of-sight projection through that sphere.

## What counts as done for a new feature

A feature is not done until it has:

1. a public API with explicit typed parameter containers;
2. a docstring specifying physical units and differentiability limitations;
3. a small runnable example or an update to an existing example;
4. finite-gradient tests;
5. shape tests;
6. no non-JAX operations inside differentiable kernels;
7. no unreviewed changes to physical conventions.

## Near-term roadmap

### Stage 1: NFW one-halo painter

- Keep improving `nfw_density` and `nfw_projected_surface_density`.
- Add analytic/semianalytic validation tests for enclosed mass and projected profile behaviour.
- Validate hard-truncated density and projected profiles against numerical
  enclosed-mass and line-of-sight references.
- Add chunked pixel painting for PLC maps.

### Stage 2: PINOCCHIO adapters

- Add explicit readers for the user's PINOCCHIO PLC and box catalogue formats.
- Keep readers in `geppetto.io`.
- Return `HaloCatalog` or `LightconeHaloCatalog` objects with JAX arrays.
- Do not import reader dependencies in `geppetto.__init__` unless they are hard dependencies.

### Stage 3: Aricò-style baryonification

Implement baryonification as profile composition, not as a separate painter. The painter should not care whether the supplied profile is NFW-only or baryonified.

Target components:

- collisionless dark matter response;
- bound gas profile;
- ejected gas profile;
- central galaxy/stellar component;
- mass compensation radius;
- projected and 3D kernels;
- parameters exposed for SBI/calibration.

The implementation should cite the exact selected Aricò et al. model in documentation and tests. Do not add a partial baryonic model with physical-looking constants unless it is tied to a chosen equation set.

### Stage 4: Map-level integration

- Accept HEALPix pixel vectors or map chunks from PINOCCHIO.
- Return arrays in the same ordering as the supplied pixel vectors.
- Keep HEALPix index generation outside core JAX kernels.
- Add optional conversion from surface density to mass per pixel or shell-density contribution when the map-level convention is fixed.

## Preferred API style

Use small pure functions:

```python
rho = density_at_points(points, catalog, cosmology, concentration_params, profile_params)
sigma = paint_lightcone_surface_density(pixel_unit_vectors, lightcone_catalog, ...)
grid = paint_box_density_grid(catalog, box_size, nmesh, ...)
```

Avoid stateful objects like:

```python
painter = Painter(...)
painter.load_file(...)
painter.paint_in_place(...)
```

Stateful orchestration can be added later outside the differentiable core if needed.

## Testing commands

From the repository root:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Important exclusions

- Do not implement cosmological distances in the core until the required convention is fixed. Accept distances from PINOCCHIO or a separate cosmology layer.
- Do not add hydro simulation calibration tables to the repository without checking licensing and provenance.
- Do not assume HEALPix `RING` or `NESTED` ordering inside the painter. GEPPETTO operates on pixel vectors supplied by the caller.
- Do not silently combine the one-halo output with PINOCCHIO's two-halo map. Return the one-halo contribution explicitly.
