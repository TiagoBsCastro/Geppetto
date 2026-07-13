"""Differentiable matter-painting kernels."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from geppetto.catalog import HaloCatalog, LightconeHaloCatalog, LightconeSparseStencil
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology, rho_mean_comoving
from geppetto.geometry import (
    box_grid_positions,
    pairwise_radius,
    transverse_distance_from_unit_vectors,
)
from geppetto.profiles import (
    DEFAULT_NFW_PROFILE_PARAMS,
    NFWProfileParams,
    TabulatedProjectedProfileParams,
    nfw_density,
    nfw_projected_surface_density,
    tabulated_projected_surface_density,
)
from geppetto.types import Array

DEFAULT_COSMOLOGY = Cosmology()
DEFAULT_CONCENTRATION_PARAMS = ConcentrationParams()


def _apply_sparse_pair_weight(contribution: Array, stencil: LightconeSparseStencil) -> Array:
    if stencil.pair_weight is None:
        return contribution
    return contribution * jnp.asarray(stencil.pair_weight, dtype=contribution.dtype)


def _density_from_catalog(
    points: Array,
    catalog: HaloCatalog,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams,
    periodic_box_size: float | None,
    halo_weight: Array | None = None,
) -> Array:
    radius = pairwise_radius(points, catalog.position, periodic_box_size)
    rho = nfw_density(
        radius,
        catalog.mass[None, :],
        catalog.redshift[None, :],
        cosmology,
        concentration_params,
        profile_params,
    )
    if halo_weight is not None:
        rho = rho * halo_weight[None, :]
    return jnp.sum(rho, axis=1)


def density_at_points(
    points: Array,
    catalog: HaloCatalog,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    periodic_box_size: float | None = None,
    as_delta: bool = False,
) -> Array:
    """Paint the one-halo 3D density at arbitrary target points.

    This is the most general GEPPETTO primitive. It is differentiable with
    respect to masses, redshifts, halo positions, target positions, cosmology
    parameters and profile/concentration parameters, up to the chosen smoothing
    and any optional periodic wrapping.
    """

    rho = _density_from_catalog(
        points, catalog, cosmology, concentration_params, profile_params, periodic_box_size
    )
    if as_delta:
        return rho / rho_mean_comoving(cosmology) - 1.0
    return rho


def _pad_catalog_for_chunks(catalog: HaloCatalog, chunk_size: int) -> tuple[HaloCatalog, Array, int]:
    n_halo = catalog.mass.shape[0]
    n_chunk = int(math.ceil(n_halo / chunk_size))
    n_pad = n_chunk * chunk_size - n_halo

    valid = jnp.concatenate([jnp.ones(n_halo), jnp.zeros(n_pad)])
    position = jnp.pad(catalog.position, ((0, n_pad), (0, 0)))
    # Use benign padded mass/redshift values and mask them out after evaluation.
    mass = jnp.pad(catalog.mass, (0, n_pad), constant_values=1.0)
    redshift = jnp.pad(catalog.redshift, (0, n_pad), constant_values=0.0)
    return HaloCatalog(position=position, mass=mass, redshift=redshift), valid, n_chunk


def density_at_points_chunked(
    points: Array,
    catalog: HaloCatalog,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    periodic_box_size: float | None = None,
    as_delta: bool = False,
    chunk_size: int = 1024,
) -> Array:
    """Chunked version of :func:`density_at_points` for many haloes.

    The chunk dimension is static, which makes this function suitable for
    ``jax.jit`` once the catalogue size and chunk size are fixed.
    """

    padded, valid, n_chunk = _pad_catalog_for_chunks(catalog, chunk_size)
    pos_chunks = padded.position.reshape(n_chunk, chunk_size, 3)
    mass_chunks = padded.mass.reshape(n_chunk, chunk_size)
    redshift_chunks = padded.redshift.reshape(n_chunk, chunk_size)
    valid_chunks = valid.reshape(n_chunk, chunk_size)

    def body(carry: Array, chunk: tuple[Array, Array, Array, Array]) -> tuple[Array, None]:
        pos, mass, z, weight = chunk
        partial_catalog = HaloCatalog(position=pos, mass=mass, redshift=z)
        partial = _density_from_catalog(
            points,
            partial_catalog,
            cosmology,
            concentration_params,
            profile_params,
            periodic_box_size,
            halo_weight=weight,
        )
        return carry + partial, None

    rho0 = jnp.zeros(points.shape[0], dtype=points.dtype)
    rho, _ = jax.lax.scan(body, rho0, (pos_chunks, mass_chunks, redshift_chunks, valid_chunks))
    if as_delta:
        return rho / rho_mean_comoving(cosmology) - 1.0
    return rho


def paint_box_density_grid(
    catalog: HaloCatalog,
    box_size: float,
    nmesh: int,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    periodic: bool = True,
    as_delta: bool = False,
    chunk_size: int | None = None,
) -> Array:
    """Paint a periodic comoving-box density grid from a halo catalogue.

    Returns an array with shape ``(nmesh, nmesh, nmesh)``.
    """

    points = box_grid_positions(box_size, nmesh)
    periodic_box_size = box_size if periodic else None
    if chunk_size is None:
        rho = density_at_points(
            points,
            catalog,
            cosmology,
            concentration_params,
            profile_params,
            periodic_box_size=periodic_box_size,
            as_delta=as_delta,
        )
    else:
        rho = density_at_points_chunked(
            points,
            catalog,
            cosmology,
            concentration_params,
            profile_params,
            periodic_box_size=periodic_box_size,
            as_delta=as_delta,
            chunk_size=chunk_size,
        )
    return rho.reshape((nmesh, nmesh, nmesh))


def paint_lightcone_surface_density(
    pixel_unit_vectors: Array,
    catalog: LightconeHaloCatalog,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    pixel_area_sr: float | None = None,
    return_mass_per_pixel: bool = False,
    chunk_size: int | None = None,
) -> Array:
    """Paint the one-halo projected surface density on lightcone pixels.

    Parameters
    ----------
    pixel_unit_vectors:
        Unit vectors of the target angular pixels, shape ``(n_pix, 3)``. This
        keeps HEALPix indexing outside the differentiable core. Use ``healpy`` or
        your PINOCCHIO map machinery to provide these vectors.
    catalog:
        Lightcone halo catalogue.
    pixel_area_sr:
        Pixel solid angle. Required only when ``return_mass_per_pixel=True``.
    return_mass_per_pixel:
        If true, convert surface density to approximate projected mass per pixel
        using ``Sigma * chi_h**2 * pixel_area_sr`` for each halo contribution.
    chunk_size:
        Optional static halo chunk size.
    """

    if return_mass_per_pixel and pixel_area_sr is None:
        raise ValueError("pixel_area_sr is required when return_mass_per_pixel=True")

    def contribution(lightcone_chunk: LightconeHaloCatalog, weight: Array | None = None) -> Array:
        r_perp = transverse_distance_from_unit_vectors(
            pixel_unit_vectors, lightcone_chunk.unit_vector, lightcone_chunk.chi
        )
        sigma = nfw_projected_surface_density(
            r_perp,
            lightcone_chunk.mass[None, :],
            lightcone_chunk.redshift[None, :],
            cosmology,
            concentration_params,
            profile_params,
        )
        if return_mass_per_pixel:
            sigma = sigma * (lightcone_chunk.chi[None, :] ** 2) * pixel_area_sr
        if weight is not None:
            sigma = sigma * weight[None, :]
        return jnp.sum(sigma, axis=1)

    if chunk_size is None:
        return contribution(catalog)

    halo_catalog = HaloCatalog(position=catalog.unit_vector, mass=catalog.mass, redshift=catalog.redshift)
    padded, valid, n_chunk = _pad_catalog_for_chunks(halo_catalog, chunk_size)
    chi = jnp.pad(catalog.chi, (0, n_chunk * chunk_size - catalog.chi.shape[0]), constant_values=1.0)
    uv_chunks = padded.position.reshape(n_chunk, chunk_size, 3)
    chi_chunks = chi.reshape(n_chunk, chunk_size)
    mass_chunks = padded.mass.reshape(n_chunk, chunk_size)
    redshift_chunks = padded.redshift.reshape(n_chunk, chunk_size)
    valid_chunks = valid.reshape(n_chunk, chunk_size)

    def body(carry: Array, chunk: tuple[Array, Array, Array, Array, Array]) -> tuple[Array, None]:
        uv, chi_i, mass, z, weight = chunk
        partial_catalog = LightconeHaloCatalog(unit_vector=uv, chi=chi_i, mass=mass, redshift=z)
        return carry + contribution(partial_catalog, weight), None

    out0 = jnp.zeros(pixel_unit_vectors.shape[0], dtype=pixel_unit_vectors.dtype)
    out, _ = jax.lax.scan(body, out0, (uv_chunks, chi_chunks, mass_chunks, redshift_chunks, valid_chunks))
    return out


def paint_lightcone_surface_density_sparse(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    pixel_area_sr: float | None = None,
    return_mass_per_pixel: bool = False,
) -> Array:
    """Paint one-halo projected surface density from a sparse halo-pixel stencil.

    Parameters
    ----------
    stencil:
        Precomputed sparse halo-pixel geometry. ``stencil.r_perp`` is in
        comoving ``Mpc/h`` and ``stencil.pix_id`` indexes the output map.
        Stencil construction, including any HEALPix indexing and fixed ``Rmax``
        cuts, belongs outside this differentiable JAX kernel.
    catalog:
        Lightcone halo catalogue. Masses are ``Msun/h`` and distances are
        comoving ``Mpc/h``.
    pixel_area_sr:
        Pixel solid angle. Required only when ``return_mass_per_pixel=True``.
    return_mass_per_pixel:
        If true, convert each pair contribution to approximate projected mass
        per pixel using ``Sigma * chi_h**2 * pixel_area_sr`` before scatter-add.

    Returns
    -------
    Array
        One-dimensional map with shape ``(stencil.n_pix,)``.

    Notes
    -----
    The sparse map is differentiable with respect to halo/profile quantities and
    profile/concentration parameters. The stencil geometry and retained pair set
    are fixed inputs and are not differentiated. Use
    ``validate_lightcone_sparse_stencil`` before entering JIT-compiled paths
    when stencils are manually constructed.
    """

    if return_mass_per_pixel and pixel_area_sr is None:
        raise ValueError("pixel_area_sr is required when return_mass_per_pixel=True")

    halo_id = jnp.asarray(stencil.halo_id, dtype=jnp.int32)
    pix_id = jnp.asarray(stencil.pix_id, dtype=jnp.int32)
    mass = catalog.mass[halo_id]
    redshift = catalog.redshift[halo_id]
    sigma = nfw_projected_surface_density(
        stencil.r_perp,
        mass,
        redshift,
        cosmology,
        concentration_params,
        profile_params,
    )
    if return_mass_per_pixel:
        chi = catalog.chi[halo_id]
        sigma = sigma * (chi**2) * pixel_area_sr
    sigma = _apply_sparse_pair_weight(sigma, stencil)

    return jnp.zeros((stencil.n_pix,), dtype=sigma.dtype).at[pix_id].add(sigma)


def _rmax_for_sparse_pairs(rmax_mpc_h: Array | float, halo_id: Array) -> Array:
    rmax = jnp.asarray(rmax_mpc_h)
    if rmax.ndim == 0:
        return rmax
    return rmax[halo_id]


def paint_lightcone_surface_density_tabulated_sparse(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    rmax_mpc_h: Array | float,
    profile_params: TabulatedProjectedProfileParams,
    pixel_area_sr: float | None = None,
    return_mass_per_pixel: bool = False,
) -> Array:
    """Paint sparse lightcone surface density from a tabulated projected profile.

    Parameters
    ----------
    stencil:
        Precomputed sparse halo-pixel geometry. ``stencil.r_perp`` is in
        comoving ``Mpc/h`` and ``stencil.pix_id`` indexes the output map.
    catalog:
        Lightcone halo catalogue with distances in comoving ``Mpc/h`` and masses
        in ``Msun/h``.
    rmax_mpc_h:
        Fixed projected support radius in comoving ``Mpc/h``. May be scalar or
        per-halo with shape ``(n_halo,)``. It is treated as non-differentiable
        geometry inside the tabulated profile kernel.
    profile_params:
        Shared dimensionless projected-profile template. The painter is
        differentiable with respect to ``profile_params.log_shape``.
    pixel_area_sr:
        Pixel solid angle. Required only when ``return_mass_per_pixel=True``.
    return_mass_per_pixel:
        If true, convert each pair contribution to projected mass per pixel
        using ``Sigma * chi_h**2 * pixel_area_sr`` before scatter-add.

    Returns
    -------
    Array
        One-dimensional map with shape ``(stencil.n_pix,)``.

    Notes
    -----
    HEALPix indexing, stencil construction, and stencil validation belong
    outside this JAX kernel. Use ``validate_lightcone_sparse_stencil`` before
    JIT-compiled paths when stencils are manually constructed. Use
    ``validate_tabulated_projected_profile_params`` outside JAX paths for
    manually constructed tabulated profile parameters.
    """

    if return_mass_per_pixel and pixel_area_sr is None:
        raise ValueError("pixel_area_sr is required when return_mass_per_pixel=True")

    halo_id = jnp.asarray(stencil.halo_id, dtype=jnp.int32)
    pix_id = jnp.asarray(stencil.pix_id, dtype=jnp.int32)
    mass = catalog.mass[halo_id]
    rmax = _rmax_for_sparse_pairs(rmax_mpc_h, halo_id)
    sigma = tabulated_projected_surface_density(
        stencil.r_perp,
        mass,
        rmax,
        profile_params,
    )
    if return_mass_per_pixel:
        chi = catalog.chi[halo_id]
        sigma = sigma * (chi**2) * pixel_area_sr
    sigma = _apply_sparse_pair_weight(sigma, stencil)

    return jnp.zeros((stencil.n_pix,), dtype=sigma.dtype).at[pix_id].add(sigma)


def paint_lightcone_particle_count_map_tabulated_sparse(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    rmax_mpc_h: Array | float,
    profile_params: TabulatedProjectedProfileParams,
    particle_mass_msun_h: float,
    pixel_area_sr: float,
) -> Array:
    """Paint a sparse tabulated one-halo map in PINOCCHIO count units.

    Parameters
    ----------
    stencil:
        Precomputed sparse halo-pixel geometry. ``stencil.r_perp`` is in
        comoving ``Mpc/h`` and ``stencil.pix_id`` indexes the output map.
    catalog:
        Lightcone halo catalogue with distances in comoving ``Mpc/h`` and masses
        in ``Msun/h``.
    rmax_mpc_h:
        Fixed projected support radius in comoving ``Mpc/h``. May be scalar or
        per-halo with shape ``(n_halo,)`` and is not differentiated.
    profile_params:
        Shared dimensionless projected-profile template.
    particle_mass_msun_h:
        PINOCCHIO particle mass in ``Msun/h``.
    pixel_area_sr:
        Pixel solid angle in steradians.

    Notes
    -----
    The returned map is projected one-halo mass per pixel divided by
    ``particle_mass_msun_h``. ``particle_mass_msun_h`` and ``pixel_area_sr`` are
    Python-side scalar checks and should remain ordinary Python floats when this
    wrapper is directly JIT-compiled.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if pixel_area_sr <= 0.0:
        raise ValueError("pixel_area_sr must be positive")

    mass_per_pixel = paint_lightcone_surface_density_tabulated_sparse(
        stencil,
        catalog,
        rmax_mpc_h,
        profile_params,
        pixel_area_sr=pixel_area_sr,
        return_mass_per_pixel=True,
    )
    return mass_per_pixel / particle_mass_msun_h


def paint_lightcone_particle_count_map_sparse(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    particle_mass_msun_h: float,
    pixel_area_sr: float,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> Array:
    """Paint a sparse count-equivalent one-halo lightcone map.

    Parameters
    ----------
    stencil:
        Precomputed sparse halo-pixel geometry. ``stencil.r_perp`` is in
        comoving ``Mpc/h`` and ``stencil.pix_id`` indexes the output map.
    catalog:
        Lightcone halo catalogue with distances in comoving ``Mpc/h`` and masses
        in ``Msun/h``.
    particle_mass_msun_h:
        PINOCCHIO particle mass in ``Msun/h``. The returned map is projected
        one-halo mass per pixel divided by this value, matching
        :func:`paint_lightcone_particle_count_map`.
    pixel_area_sr:
        Pixel solid angle in steradians.

    Notes
    -----
    The result is differentiable with respect to halo/profile quantities and
    profile/concentration parameters. Pixel identities, HEALPix geometry, and
    the retained stencil pair set are fixed inputs outside this JAX kernel.
    ``particle_mass_msun_h`` and ``pixel_area_sr`` are Python-side scalar checks
    and should remain ordinary Python floats when this wrapper is directly
    JIT-compiled.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if pixel_area_sr <= 0.0:
        raise ValueError("pixel_area_sr must be positive")

    mass_per_pixel = paint_lightcone_surface_density_sparse(
        stencil,
        catalog,
        cosmology=cosmology,
        concentration_params=concentration_params,
        profile_params=profile_params,
        pixel_area_sr=pixel_area_sr,
        return_mass_per_pixel=True,
    )
    return mass_per_pixel / particle_mass_msun_h


def paint_lightcone_particle_count_map(
    pixel_unit_vectors: Array,
    catalog: LightconeHaloCatalog,
    particle_mass_msun_h: float,
    pixel_area_sr: float,
    cosmology: Cosmology = DEFAULT_COSMOLOGY,
    concentration_params: ConcentrationParams = DEFAULT_CONCENTRATION_PARAMS,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    chunk_size: int | None = None,
) -> Array:
    """Paint a count-equivalent one-halo mass collector on lightcone pixels.

    Parameters
    ----------
    pixel_unit_vectors:
        Unit vectors of the target angular pixels, shape ``(n_pix, 3)``.
        HEALPix pixel-index generation belongs outside this JAX kernel.
    catalog:
        Lightcone halo catalogue with distances in comoving ``Mpc/h`` and masses
        in ``Msun/h``.
    particle_mass_msun_h:
        PINOCCHIO particle mass in ``Msun/h``. The returned map is projected
        one-halo mass per pixel divided by this value, so it can be interpreted
        as a particle-count-equivalent mass collector.
    pixel_area_sr:
        Pixel solid angle in steradians.
    chunk_size:
        Optional static halo chunk size.

    Notes
    -----
    The returned values are differentiable with respect to halo/profile
    quantities and profile parameters. Pixel identities and HEALPix geometry are
    fixed inputs and are intentionally outside the differentiable core.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if pixel_area_sr <= 0.0:
        raise ValueError("pixel_area_sr must be positive")

    mass_per_pixel = paint_lightcone_surface_density(
        pixel_unit_vectors,
        catalog,
        cosmology=cosmology,
        concentration_params=concentration_params,
        profile_params=profile_params,
        pixel_area_sr=pixel_area_sr,
        return_mass_per_pixel=True,
        chunk_size=chunk_size,
    )
    return mass_per_pixel / particle_mass_msun_h
