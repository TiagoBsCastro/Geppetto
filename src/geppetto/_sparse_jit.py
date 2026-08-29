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

DERIVATIVE_COMPARISON_SUM_STAT_NAMES = (
    "autodiff_squared_norm",
    "finite_difference_squared_norm",
    "residual_squared_norm",
    "autodiff_finite_difference_dot",
    "autodiff_sum",
    "finite_difference_sum",
)
DERIVATIVE_COMPARISON_MAX_STAT_NAMES = (
    "maximum_absolute_autodiff",
    "maximum_absolute_finite_difference",
    "maximum_absolute_residual",
)


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


def _particle_count_map_concentration_central_difference(
    stencil: AdaptiveLightconeStencil,
    catalog: LightconeHaloCatalog,
    theta: Array,
    direction: Array,
    step: Array,
    particle_mass_msun_h: Array,
    cosmology: Cosmology,
    concentration_mass_pivot: Array,
    overdensity: Array,
    *,
    overdensity_mode: str,
    reference_density: str,
    sample_chunk_size: int,
) -> tuple[Array, Array, Array]:
    """Return a central-difference map, global sum, and invalid counts."""

    offset = step * direction
    plus = _particle_count_map_from_concentration(
        stencil,
        catalog,
        theta + offset,
        particle_mass_msun_h,
        cosmology,
        concentration_mass_pivot,
        overdensity,
        overdensity_mode=overdensity_mode,
        reference_density=reference_density,
        sample_chunk_size=sample_chunk_size,
    )
    minus = _particle_count_map_from_concentration(
        stencil,
        catalog,
        theta - offset,
        particle_mass_msun_h,
        cosmology,
        concentration_mass_pivot,
        overdensity,
        overdensity_mode=overdensity_mode,
        reference_density=reference_density,
        sample_chunk_size=sample_chunk_size,
    )
    denominator = 2.0 * step
    invalid_counts = jnp.stack(
        (
            jnp.sum(plus.invalid_normalization, dtype=jnp.int32),
            jnp.sum(minus.invalid_normalization, dtype=jnp.int32),
        )
    )
    return (
        (plus.particle_counts - minus.particle_counts) / denominator,
        (
            plus.assigned_global_particle_count
            - minus.assigned_global_particle_count
        )
        / denominator,
        invalid_counts,
    )


def _derivative_comparison_statistics(
    autodiff: Array,
    finite_difference: Array,
) -> tuple[Array, Array]:
    """Return additive and maximum statistics for two derivative maps."""

    residual = finite_difference - autodiff
    sum_statistics = jnp.stack(
        (
            jnp.sum(autodiff * autodiff),
            jnp.sum(finite_difference * finite_difference),
            jnp.sum(residual * residual),
            jnp.sum(autodiff * finite_difference),
            jnp.sum(autodiff),
            jnp.sum(finite_difference),
        )
    )
    max_statistics = jnp.stack(
        (
            jnp.max(jnp.abs(autodiff)),
            jnp.max(jnp.abs(finite_difference)),
            jnp.max(jnp.abs(residual)),
        )
    )
    return sum_statistics, max_statistics


paint_nfw_particle_count_map_sparse_jit = jax.jit(
    _particle_count_map_from_concentration,
    static_argnames=("overdensity_mode", "reference_density", "sample_chunk_size"),
)
paint_nfw_particle_count_map_and_concentration_jvps_jit = jax.jit(
    _particle_count_map_and_concentration_jvps,
    static_argnames=("overdensity_mode", "reference_density", "sample_chunk_size"),
)
paint_nfw_particle_count_map_concentration_central_difference_jit = jax.jit(
    _particle_count_map_concentration_central_difference,
    static_argnames=("overdensity_mode", "reference_density", "sample_chunk_size"),
)
derivative_comparison_statistics_jit = jax.jit(
    _derivative_comparison_statistics,
)
