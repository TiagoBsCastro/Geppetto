"""Differentiable halo-profile prescriptions.

The first production profile is NFW. Additional components for baryonification
should be added as separate composable profile transforms rather than hidden
inside the NFW implementation.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax.nn as jnn
import jax.numpy as jnp

from geppetto.concentration import ConcentrationParams, concentration_power_law
from geppetto.cosmology import Cosmology, halo_radius_delta_comoving
from geppetto.types import Array


class NFWProfileParams(NamedTuple):
    """Parameters controlling the NFW profile normalization and truncation."""

    overdensity: float = 200.0
    reference_density: Literal["critical", "mean"] = "critical"
    smooth_truncation: bool = True
    truncation_width_fraction: float = 0.05
    r_softening_fraction: float = 1.0e-4


DEFAULT_NFW_PROFILE_PARAMS = NFWProfileParams()


def nfw_shape_function(c: Array) -> Array:
    """Return ``ln(1+c) - c/(1+c)``."""

    return jnp.log1p(c) - c / (1.0 + c)


def nfw_scale_radius_and_density(
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> tuple[Array, Array, Array, Array]:
    """Return ``(r_delta, concentration, r_s, rho_s)`` in comoving units.

    The profile integrates to ``mass`` inside ``r_delta`` before any optional
    smooth taper is applied.
    """

    c = concentration_power_law(mass, redshift, concentration_params)
    r_delta = halo_radius_delta_comoving(
        mass,
        redshift,
        cosmology,
        overdensity=profile_params.overdensity,
        reference_density=profile_params.reference_density,
    )
    r_s = r_delta / c
    rho_s = mass / (4.0 * jnp.pi * r_s**3 * nfw_shape_function(c))
    return r_delta, c, r_s, rho_s


def smooth_taper(r: Array, r_delta: Array, width: Array) -> Array:
    """Smoothly suppress the profile outside ``r_delta``.

    This avoids a hard non-differentiable truncation at the halo boundary. The
    taper is not exactly mass conserving; explicit mass-conserving truncation can
    be added as a later profile prescription.
    """

    return jnn.sigmoid(-(r - r_delta) / width)


def nfw_density(
    r: Array,
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> Array:
    """Evaluate the 3D NFW density in comoving ``(Msun/h)/(Mpc/h)^3`` units."""

    r_delta, _, r_s, rho_s = nfw_scale_radius_and_density(
        mass, redshift, cosmology, concentration_params, profile_params
    )
    r_safe = jnp.sqrt(r**2 + (profile_params.r_softening_fraction * r_s) ** 2)
    x = r_safe / r_s
    rho = rho_s / (x * (1.0 + x) ** 2)

    if profile_params.smooth_truncation:
        width = jnp.maximum(profile_params.truncation_width_fraction * r_delta, 1.0e-12)
        return rho * smooth_taper(r_safe, r_delta, width)
    return jnp.where(r_safe <= r_delta, rho, 0.0)


def _projected_nfw_kernel(x: Array) -> Array:
    """Dimensionless projected NFW kernel for Sigma(R) = 2 rho_s r_s F(x).

    The expression is evaluated with a small exclusion around x=1 to avoid the
    removable singularity. It is differentiable almost everywhere and stable for
    normal map-painting usage.
    """

    eps = 1.0e-5
    x_safe_low = jnp.minimum(x, 1.0 - eps)
    x_safe_high = jnp.maximum(x, 1.0 + eps)

    low_arg = jnp.sqrt((1.0 - x_safe_low) / (1.0 + x_safe_low))
    low = (1.0 - 2.0 / jnp.sqrt(1.0 - x_safe_low**2) * jnp.arctanh(low_arg)) / (x_safe_low**2 - 1.0)

    high_arg = jnp.sqrt((x_safe_high - 1.0) / (1.0 + x_safe_high))
    high = (1.0 - 2.0 / jnp.sqrt(x_safe_high**2 - 1.0) * jnp.arctan(high_arg)) / (x_safe_high**2 - 1.0)

    near = jnp.ones_like(x) / 3.0
    return jnp.where(x < 1.0 - eps, low, jnp.where(x > 1.0 + eps, high, near))


def nfw_projected_surface_density(
    r_perp: Array,
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> Array:
    """Projected NFW surface density in comoving ``(Msun/h)/(Mpc/h)^2`` units."""

    r_delta, _, r_s, rho_s = nfw_scale_radius_and_density(
        mass, redshift, cosmology, concentration_params, profile_params
    )
    r_safe = jnp.sqrt(r_perp**2 + (profile_params.r_softening_fraction * r_s) ** 2)
    sigma = 2.0 * rho_s * r_s * _projected_nfw_kernel(r_safe / r_s)
    if profile_params.smooth_truncation:
        width = jnp.maximum(profile_params.truncation_width_fraction * r_delta, 1.0e-12)
        return sigma * smooth_taper(r_safe, r_delta, width)
    return jnp.where(r_safe <= r_delta, sigma, 0.0)
