"""Differentiable halo-profile prescriptions.

The first production profile is NFW. Additional components for baryonification
should be added as separate composable profile transforms rather than hidden
inside the NFW implementation.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax import lax

from geppetto.concentration import ConcentrationParams, concentration_power_law
from geppetto.cosmology import (
    Cosmology,
    bryan_norman_virial_overdensity,
    halo_radius_delta_comoving,
)
from geppetto.types import Array


class NFWProfileParams(NamedTuple):
    """Parameters controlling the hard-truncated NFW profile.

    ``overdensity`` is the constant spherical overdensity used when
    ``overdensity_mode='constant'``. ``reference_density`` selects physical
    critical or mean-matter density. In ``'bryan_norman'`` mode, the overdensity
    is evaluated from halo redshift and ``overdensity`` is ignored.
    """

    overdensity: float = 200.0
    reference_density: Literal["critical", "mean"] = "critical"
    r_softening_fraction: float = 1.0e-4
    overdensity_mode: Literal["constant", "bryan_norman"] = "constant"


DEFAULT_NFW_PROFILE_PARAMS = NFWProfileParams()


class TabulatedProjectedProfileParams(NamedTuple):
    """Dimensionless projected-profile template parameters.

    Parameters
    ----------
    x:
        Dimensionless projected radius grid ``R / Rmax``, shape
        ``(n_radius,)``. The JAX kernel assumes this grid is finite, increasing,
        and covers ``[0, 1]``.
    log_shape:
        Unconstrained log projected-profile shape values at ``x``, shape
        ``(n_radius,)``. The kernel exponentiates these values and normalizes
        the resulting template so the projected mass inside fixed ``Rmax``
        equals the supplied halo mass.
    """

    x: Array
    log_shape: Array


def nfw_shape_function(c: Array) -> Array:
    """Return ``ln(1+c) - c/(1+c)``."""

    return jnp.log1p(c) - c / (1.0 + c)


def nfw_halo_overdensity(
    redshift: Array,
    cosmology: Cosmology,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> Array:
    """Return the NFW spherical overdensity at each halo redshift.

    The result is dimensionless and may be scalar for a constant definition or
    halo-shaped for the Bryan--Norman virial definition. This function is
    differentiable with respect to redshift and cosmological parameters.
    """

    if profile_params.overdensity_mode == "constant":
        return jnp.asarray(profile_params.overdensity)
    if profile_params.overdensity_mode == "bryan_norman":
        return bryan_norman_virial_overdensity(
            redshift,
            cosmology,
            reference_density=profile_params.reference_density,
        )
    raise ValueError(
        f"Unknown overdensity_mode={profile_params.overdensity_mode!r}"
    )


def nfw_scale_radius_and_density(
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> tuple[Array, Array, Array, Array]:
    """Return ``(r_delta, concentration, r_s, rho_s)`` in comoving units.

    The unsoftened profile integrates to ``mass`` inside ``r_delta``.
    """

    c = concentration_power_law(mass, redshift, concentration_params)
    overdensity = nfw_halo_overdensity(redshift, cosmology, profile_params)
    r_delta = halo_radius_delta_comoving(
        mass,
        redshift,
        cosmology,
        overdensity=overdensity,
        reference_density=profile_params.reference_density,
    )
    r_s = r_delta / c
    rho_s = mass / (4.0 * jnp.pi * r_s**3 * nfw_shape_function(c))
    return r_delta, c, r_s, rho_s


def nfw_density(
    r: Array,
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> Array:
    """Evaluate hard-truncated 3D NFW density in comoving mass-density units.

    Radii are comoving ``Mpc/h`` and the result is in
    ``(Msun/h)/(Mpc/h)^3``. The support test uses the original radius, so the
    fixed ``r_delta`` boundary is independent of concentration. The explicit
    central-radius softening regularizes the NFW cusp below
    ``r_softening_fraction * r_s``.
    """

    r_delta, _, r_s, rho_s = nfw_scale_radius_and_density(
        mass, redshift, cosmology, concentration_params, profile_params
    )
    r_safe = jnp.sqrt(r**2 + (profile_params.r_softening_fraction * r_s) ** 2)
    x = r_safe / r_s
    rho = rho_s / (x * (1.0 + x) ** 2)
    return jnp.where(r <= r_delta, rho, 0.0)


def _hard_truncated_projected_nfw_kernel(x: Array, concentration: Array) -> Array:
    """Return the finite line-of-sight NFW kernel inside ``r_delta``.

    The result ``F(x, c)`` obeys ``Sigma = 2 * rho_s * r_s * F`` and projects
    the three-dimensional NFW density only through the sphere ``r <= r_delta``.
    The branch expressions are equations (6)--(7) of Hamana et al. (2004),
    following Takada & Jain (2003). A narrow linear bridge around ``x=1``
    avoids cancellation in the removable singularity while retaining finite
    concentration derivatives.
    """

    dtype = jnp.result_type(x, concentration)
    x = jnp.asarray(x, dtype=dtype)
    concentration = jnp.asarray(concentration, dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    near_width = jnp.asarray(1.0e-2 if dtype == jnp.float32 else 1.0e-3, dtype=dtype)

    c_safe = jnp.maximum(concentration, tiny)
    # All branch expressions remain finite even where their result is masked.
    x_inside = jnp.clip(x, tiny, c_safe * (1.0 - 4.0 * jnp.finfo(dtype).eps))

    def low_branch(x_value: Array) -> Array:
        one_minus_x2 = jnp.maximum(1.0 - x_value**2, tiny)
        root = jnp.sqrt(jnp.maximum(c_safe**2 - x_value**2, 0.0))
        argument = (x_value**2 + c_safe) / (x_value * (1.0 + c_safe))
        argument = jnp.maximum(argument, 1.0)
        return (
            -root / (one_minus_x2 * (1.0 + c_safe))
            + jnp.arccosh(argument) / one_minus_x2**1.5
        )

    def high_branch(x_value: Array) -> Array:
        x2_minus_one = jnp.maximum(x_value**2 - 1.0, tiny)
        root = jnp.sqrt(jnp.maximum(c_safe**2 - x_value**2, 0.0))
        argument = (x_value**2 + c_safe) / (x_value * (1.0 + c_safe))
        argument = jnp.clip(argument, -1.0, 1.0)
        return (
            root / (x2_minus_one * (1.0 + c_safe))
            - jnp.arccos(argument) / x2_minus_one**1.5
        )

    c_near = jnp.maximum(c_safe, 1.0 + near_width)
    center = (
        jnp.sqrt(jnp.maximum(c_near**2 - 1.0, 0.0))
        / (3.0 * (1.0 + c_near))
        * (1.0 + 1.0 / (1.0 + c_near))
    )
    x_minus = jnp.ones_like(x_inside) - near_width
    x_plus = jnp.ones_like(x_inside) + near_width
    value_minus = low_branch(x_minus)
    value_plus = high_branch(jnp.minimum(x_plus, c_near * (1.0 - 4.0 * jnp.finfo(dtype).eps)))
    near_low = value_minus + (x_inside - x_minus) * (center - value_minus) / near_width
    near_high = center + (x_inside - 1.0) * (value_plus - center) / near_width

    low = low_branch(jnp.minimum(x_inside, 1.0 - near_width))
    high = high_branch(jnp.maximum(x_inside, 1.0 + near_width))
    value = jnp.where(
        x_inside < 1.0 - near_width,
        low,
        jnp.where(
            x_inside < 1.0,
            near_low,
            jnp.where(x_inside <= 1.0 + near_width, near_high, high),
        ),
    )
    return jnp.where((x < concentration) & (concentration > 0.0), value, 0.0)


def nfw_projected_surface_density(
    r_perp: Array,
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
) -> Array:
    """Project a hard-truncated 3D NFW halo through its ``r_delta`` sphere.

    ``r_perp`` is a comoving projected radius in ``Mpc/h`` and the result is in
    ``(Msun/h)/(Mpc/h)^2``. The finite line-of-sight limit is
    ``sqrt(r_delta**2 - r_perp**2)``. The support test uses the original
    projected radius and is therefore independent of concentration. A tiny
    explicit central-radius softening regularizes the logarithmic projected cusp.
    """

    r_delta, concentration, r_s, rho_s = nfw_scale_radius_and_density(
        mass, redshift, cosmology, concentration_params, profile_params
    )
    r_safe = jnp.sqrt(r_perp**2 + (profile_params.r_softening_fraction * r_s) ** 2)
    kernel = _hard_truncated_projected_nfw_kernel(r_safe / r_s, concentration)
    sigma = 2.0 * rho_s * r_s * kernel
    return jnp.where(r_perp < r_delta, sigma, 0.0)


def _linear_interpolate(x_eval: Array, x_grid: Array, y_grid: Array) -> Array:
    """Evaluate a one-dimensional linear interpolant on a fixed grid."""

    x_clipped = jnp.clip(x_eval, x_grid[0], x_grid[-1])
    idx = jnp.searchsorted(x_grid, x_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, x_grid.shape[0] - 2)
    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    y0 = y_grid[idx]
    y1 = y_grid[idx + 1]
    t = (x_clipped - x0) / jnp.maximum(x1 - x0, 1.0e-30)
    return y0 + t * (y1 - y0)


def _trapezoid_integral(y: Array, x: Array) -> Array:
    """Integrate ``y(x)`` with the trapezoid rule along the only axis."""

    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


def tabulated_projected_surface_density(
    r_perp: Array,
    mass: Array,
    rmax_mpc_h: Array | float,
    profile_params: TabulatedProjectedProfileParams,
) -> Array:
    """Evaluate a mass-normalized tabulated projected profile.

    Parameters
    ----------
    r_perp:
        Projected comoving radius in ``Mpc/h``.
    mass:
        Halo mass in ``Msun/h``.
    rmax_mpc_h:
        Fixed projected support radius in comoving ``Mpc/h``. This value is
        treated as non-differentiable geometry by applying
        :func:`jax.lax.stop_gradient` inside the kernel.
    profile_params:
        Shared dimensionless template. ``log_shape`` is differentiable; ``x``
        is treated as fixed non-differentiable geometry by applying
        :func:`jax.lax.stop_gradient` inside the kernel.

    Returns
    -------
    Array
        Projected surface density in comoving ``(Msun/h)/(Mpc/h)^2`` units.

    Notes
    -----
    The projected template is normalized with
    ``2*pi*integral_0^1 x*shape(x) dx`` so the mass inside ``Rmax`` equals
    ``mass`` in the continuum. Values outside the last tabulated radius are set
    to zero. Validate manually constructed tabulated profiles outside JAX paths
    before passing them to this kernel.
    """

    x_grid = lax.stop_gradient(jnp.asarray(profile_params.x))
    shape_grid = jnp.exp(jnp.asarray(profile_params.log_shape))
    rmax = lax.stop_gradient(jnp.asarray(rmax_mpc_h))
    rmax_safe = jnp.maximum(rmax, 1.0e-30)
    x_eval = jnp.asarray(r_perp) / rmax_safe

    shape = _linear_interpolate(x_eval, x_grid, shape_grid)
    shape = jnp.where((x_eval >= x_grid[0]) & (x_eval <= x_grid[-1]), shape, 0.0)

    integral = _trapezoid_integral(x_grid * shape_grid, x_grid)
    norm = jnp.maximum(2.0 * jnp.pi * integral, 1.0e-30)
    return jnp.asarray(mass) * shape / (rmax_safe**2 * norm)
