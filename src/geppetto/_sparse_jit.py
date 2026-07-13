"""Reusable compiled kernels for the production sparse NFW workflow."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from geppetto.catalog import LightconeHaloCatalog, LightconeSparseStencil
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology
from geppetto.painters import paint_lightcone_surface_density_sparse
from geppetto.profiles import NFWProfileParams
from geppetto.types import Array


def _particle_count_map_from_concentration(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    theta: Array,
    particle_mass_msun_h: Array,
    pixel_area_sr: Array,
    cosmology: Cosmology,
    concentration_mass_pivot: Array,
    truncation_width_fraction: Array,
) -> Array:
    concentration_params = ConcentrationParams(
        amplitude=theta[0],
        mass_slope=theta[1],
        redshift_slope=theta[2],
        mass_pivot=concentration_mass_pivot,
    )
    profile_params = NFWProfileParams(
        truncation_width_fraction=truncation_width_fraction,
    )
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


def _particle_count_map_and_concentration_jvps(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    theta: Array,
    particle_mass_msun_h: Array,
    pixel_area_sr: Array,
    cosmology: Cosmology,
    concentration_mass_pivot: Array,
    truncation_width_fraction: Array,
) -> tuple[Array, Array]:
    def paint(theta_value: Array) -> Array:
        return _particle_count_map_from_concentration(
            stencil,
            catalog,
            theta_value,
            particle_mass_msun_h,
            pixel_area_sr,
            cosmology,
            concentration_mass_pivot,
            truncation_width_fraction,
        )

    particle_counts, linearized_paint = jax.linearize(paint, theta)
    basis = jnp.eye(theta.shape[0], dtype=theta.dtype)
    derivatives = jax.vmap(linearized_paint)(basis)
    return particle_counts, derivatives


paint_nfw_particle_count_map_sparse_jit = jax.jit(
    _particle_count_map_from_concentration
)
paint_nfw_particle_count_map_and_concentration_jvps_jit = jax.jit(
    _particle_count_map_and_concentration_jvps
)
