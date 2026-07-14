"""Reusable compiled kernels for the production sparse NFW workflow."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from geppetto.catalog import AdaptiveLightconeStencil, LightconeHaloCatalog
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology
from geppetto.painters import (
    AdaptiveParticlePaintResult,
    _paint_lightcone_particle_count_map_adaptive,
)
from geppetto.profiles import NFWProfileParams
from geppetto.types import Array


def _particle_count_map_from_concentration(
    stencil: AdaptiveLightconeStencil,
    catalog: LightconeHaloCatalog,
    theta: Array,
    particle_mass_msun_h: Array,
    cosmology: Cosmology,
    concentration_mass_pivot: Array,
    overdensity: Array,
    *,
    overdensity_mode: str,
    reference_density: str,
    sample_chunk_size: int,
) -> AdaptiveParticlePaintResult:
    concentration_params = ConcentrationParams(
        amplitude=theta[0],
        mass_slope=theta[1],
        redshift_slope=theta[2],
        mass_pivot=concentration_mass_pivot,
    )
    profile_params = NFWProfileParams(
        overdensity=overdensity,
        reference_density=reference_density,
        overdensity_mode=overdensity_mode,
    )
    return _paint_lightcone_particle_count_map_adaptive(
        stencil,
        catalog,
        particle_mass_msun_h,
        cosmology=cosmology,
        concentration_params=concentration_params,
        profile_params=profile_params,
        sample_chunk_size=sample_chunk_size,
    )


def _particle_count_map_and_concentration_jvps(
    stencil: AdaptiveLightconeStencil,
    catalog: LightconeHaloCatalog,
    theta: Array,
    particle_mass_msun_h: Array,
    cosmology: Cosmology,
    concentration_mass_pivot: Array,
    overdensity: Array,
    *,
    overdensity_mode: str,
    reference_density: str,
    sample_chunk_size: int,
) -> tuple[AdaptiveParticlePaintResult, Array, Array]:
    def paint(theta_value: Array) -> AdaptiveParticlePaintResult:
        return _particle_count_map_from_concentration(
            stencil,
            catalog,
            theta_value,
            particle_mass_msun_h,
            cosmology,
            concentration_mass_pivot,
            overdensity,
            overdensity_mode=overdensity_mode,
            reference_density=reference_density,
            sample_chunk_size=sample_chunk_size,
        )

    result, linearized_paint = jax.linearize(paint, theta)
    basis = jnp.eye(theta.shape[0], dtype=theta.dtype)
    derivative_results = jax.vmap(linearized_paint)(basis)
    return (
        result,
        derivative_results.particle_counts,
        derivative_results.assigned_global_particle_count,
    )


paint_nfw_particle_count_map_sparse_jit = jax.jit(
    _particle_count_map_from_concentration,
    static_argnames=("overdensity_mode", "reference_density", "sample_chunk_size"),
)
paint_nfw_particle_count_map_and_concentration_jvps_jit = jax.jit(
    _particle_count_map_and_concentration_jvps,
    static_argnames=("overdensity_mode", "reference_density", "sample_chunk_size"),
)
