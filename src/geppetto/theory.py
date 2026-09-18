"""Matter-power and angular-spectrum theory for PINOCCHIO map validation.

The differentiable part of this module includes the standard, normalized-HMF
halo model and the legacy compensated-response model. PINOCCHIO supplies the
tabulated linear spectrum, its optional scale-dependent time evolution,
measured halo mass functions, and a host-fitted halo-bias table; GEPPETTO
supplies the NFW mass definition and concentration relation used by the map
painter.

All masses are ``Msun/h``, comoving distances are ``Mpc/h``, wavenumbers are
``h/Mpc``, three-dimensional power spectra are ``(Mpc/h)^3``, and angular
power spectra are dimensionless. File parsing and HEALPix operations live
outside this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from geppetto.concentration import ConcentrationParams, concentration_power_law
from geppetto.cosmology import Cosmology, halo_radius_delta_comoving, rho_mean_comoving
from geppetto.profiles import (
    DEFAULT_NFW_PROFILE_PARAMS,
    NFWProfileParams,
    nfw_halo_overdensity,
)
from geppetto.types import Array


class LinearTheoryTable(NamedTuple):
    """PINOCCHIO background and linear-power tables in GEPPETTO units.

    ``scale_factor`` is increasing. ``chi_mpc_h`` therefore decreases from the
    high-redshift end toward zero. ``power_mpc_h3`` is the linear matter power
    spectrum at ``z=0`` and ``growth`` is normalized to unity at ``a=1``.
    """

    h: float
    omega_m0: float
    scale_factor: Array
    chi_mpc_h: Array
    omega_m: Array
    growth: Array
    k_h_mpc: Array
    power_mpc_h3: Array


class LinearPowerEvolutionTable(NamedTuple):
    """Scale-dependent PINOCCHIO linear power in GEPPETTO units.

    ``scale_factor`` and ``k_h_mpc`` are increasing. ``power_mpc_h3`` has
    shape ``(n_scale_factor, n_k)`` and is measured in ``(Mpc/h)^3``. This
    explicit pytree is optional because PINOCCHIO runs supplied with a single
    input spectrum only provide the separable growth approximation stored in
    :class:`LinearTheoryTable`.
    """

    scale_factor: Array
    k_h_mpc: Array
    power_mpc_h3: Array


class HaloMassFunctionTable(NamedTuple):
    """Measured PINOCCHIO mass functions on a common mass grid.

    ``scale_factor`` is increasing, ``log_mass_msun_h`` is the natural
    logarithm of mass in ``Msun/h``, and ``dndlnm_mpc_h3`` has shape
    ``(n_scale_factor, n_mass)`` in ``(Mpc/h)^-3``.
    """

    scale_factor: Array
    log_mass_msun_h: Array
    dndlnm_mpc_h3: Array


class HaloCountQuadrature(NamedTuple):
    """Resolved halo counts divided by box volume, with no mass extrapolation.

    ``scale_factor`` is increasing and ``log_mass_msun_h`` contains the
    natural logarithm of the native representative masses in ``Msun/h``.
    ``number_density_weight_mpc_h3`` has shape ``(n_a, n_mass)`` and units
    ``(Mpc/h)^-3``. These are integration weights, not ``dn/dlnM``: sum
    ``weight * f(mass)`` directly, without another mass-bin width.

    The count moment is exact for the input tables. Higher moments use each
    bin's representative mass; in particular the tables do not determine
    the within-bin mass variance needed for an exact one-halo self term.
    This pytree and its quadrature are independent of concentration.
    """

    scale_factor: Array
    log_mass_msun_h: Array
    number_density_weight_mpc_h3: Array


class HaloBiasTable(NamedTuple):
    """Numerical-HMF halo bias on the theory mass and scale-factor grid.

    All bias arrays have shape ``(n_scale_factor, n_mass)`` and are
    dimensionless. ``linear_bias`` is ``pbs_bias * correction``. The mass
    grid is in natural-log ``Msun/h`` and must match the corresponding
    :class:`HaloMassFunctionTable` grid.
    """

    scale_factor: Array
    log_mass_msun_h: Array
    pbs_bias: Array
    correction: Array
    linear_bias: Array


class NormalizedHaloTable(NamedTuple):
    """Quadrature of a normalized fitted HMF, including analytic tails.

    ``mass_fraction_weight`` and ``biased_mass_fraction_weight`` include
    integration weights in log mass and have shape ``(n_a, n_mass)``.
    Masses are ``Msun/h``. Tail arrays have shape ``(n_a,)`` and represent
    the complete population below the integration interval with ``u=1``.
    The negligible upper tail is included in these point-limit weights.
    Host fitting must check its error bound before using this approximation.
    Linear interpolation in scale factor preserves both closure relations.
    This immutable pytree is independent of concentration.
    """

    scale_factor: Array
    mass_msun_h: Array
    mass_fraction_weight: Array
    biased_mass_fraction_weight: Array
    tail_mass_fraction: Array
    tail_biased_mass_fraction: Array


class NormalizedHaloPower(NamedTuple):
    """Standard response and two one-halo powers, in ``(Mpc/h)^3``.

    ``response`` is dimensionless. Each field has the shape of the input k.
    The standard one-halo term has a white low-k limit; the compensated
    term uses the Lagrangian top-hat difference and has a k^4 limit.
    """

    response: Array
    one_halo_standard: Array
    one_halo_compensated: Array


class NormalizedHaloPowerTable(NamedTuple):
    """Shared profile integrals at increasing scale factor and k in h/Mpc."""

    scale_factor: Array
    k_h_mpc: Array
    response: Array
    one_halo_standard: Array
    one_halo_compensated: Array


class TwoHaloResponseTable(NamedTuple):
    """Compensated deterministic response for one radial shell.

    ``response`` has shape ``(n_scale_factor, n_k)``. Wavenumbers are in
    ``h/Mpc`` and all other arrays are dimensionless. The table is built from
    JAX operations, so interpolation remains differentiable with respect to
    concentration parameters until a host-side exact projection is requested.
    """

    scale_factor: Array
    k_h_mpc: Array
    response: Array


class QuadratureRule(NamedTuple):
    """Nodes and weights for integration over the interval ``[-1, 1]``."""

    nodes: Array
    weights: Array


class AngularPowerSpectra(NamedTuple):
    """Per-shell and count-weighted-sum angular power spectra.

    Every shell array has shape ``(n_shell, n_ell)``. Summed arrays have shape
    ``(n_ell,)``. ``linear``, ``two_halo``, and ``one_halo`` include the
    supplied HEALPix pixel window. The clustering total uses ``two_halo`` plus
    ``one_halo``; ``linear`` is diagnostic. Particle shot noise is kept white
    in the pixel-count map convention.
    """

    ell: Array
    shell_linear: Array
    shell_two_halo: Array
    shell_one_halo: Array
    shell_particle_shot_noise: Array
    shell_clustering: Array
    shell_total: Array
    summed_linear: Array
    summed_two_halo: Array
    summed_one_halo: Array
    summed_particle_shot_noise: Array
    summed_clustering: Array
    summed_total: Array
    shell_weights: Array
    shell_ell_high_ell_start: Array
    summed_ell_limber_start: Array
    ell_limber_start: Array
    high_ell_match_shell_relative_error: Array
    limber_match_summed_relative_error: Array
    shell_linear_high_ell_mode: Array


LINEAR_HIGH_ELL_LIMBER = 0
LINEAR_HIGH_ELL_FINITE_WIDTH = 1
LINEAR_EVOLUTION_INTERPOLATION_ORDER = 8


def gauss_legendre_rule(order: int) -> QuadratureRule:
    """Return a fixed Gauss--Legendre rule for non-differentiable geometry."""

    if order < 2:
        raise ValueError("quadrature order must be at least 2")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return QuadratureRule(nodes=jnp.asarray(nodes), weights=jnp.asarray(weights))


def _linear_interpolate(x: Array, x_grid: Array, values: Array) -> Array:
    """Linearly interpolate the first axis of ``values`` on an increasing grid."""

    x_clipped = jnp.clip(x, x_grid[0], x_grid[-1])
    upper = jnp.searchsorted(x_grid, x_clipped, side="right")
    upper = jnp.clip(upper, 1, x_grid.shape[0] - 1)
    lower = upper - 1
    x0 = x_grid[lower]
    x1 = x_grid[upper]
    fraction = (x_clipped - x0) / (x1 - x0)
    value0 = values[lower]
    value1 = values[upper]
    expand = (None,) * (values.ndim - 1)
    fraction = fraction[(...,) + expand]
    return value0 + fraction * (value1 - value0)


def _linear_interpolate_last_axis(x: Array, x_grid: Array, values: Array) -> Array:
    """Interpolate the last axis of ``values`` on an increasing grid."""

    coordinate = jnp.asarray(x)
    grid = jnp.asarray(x_grid)
    samples = jnp.asarray(values)
    target_shape = jnp.broadcast_shapes(coordinate.shape, samples.shape[:-1])
    coordinate = jnp.broadcast_to(coordinate, target_shape)
    samples = jnp.broadcast_to(samples, target_shape + (samples.shape[-1],))
    clipped = jnp.clip(coordinate, grid[0], grid[-1])
    upper = jnp.searchsorted(grid, clipped, side="right")
    upper = jnp.clip(upper, 1, grid.shape[0] - 1)
    lower = upper - 1
    value0 = jnp.take_along_axis(samples, lower[..., None], axis=-1)[..., 0]
    value1 = jnp.take_along_axis(samples, upper[..., None], axis=-1)[..., 0]
    fraction = (clipped - grid[lower]) / (grid[upper] - grid[lower])
    return value0 + fraction * (value1 - value0)


def _lagrange_basis(x: Array, nodes: Array) -> Array:
    """Evaluate fixed-node Lagrange basis polynomials with JAX primitives."""

    values = jnp.asarray(x)
    interpolation_nodes = jnp.asarray(nodes)
    columns = []
    for index in range(interpolation_nodes.shape[0]):
        numerator = jnp.ones_like(values)
        denominator = jnp.asarray(1.0, dtype=values.dtype)
        for other in range(interpolation_nodes.shape[0]):
            if other == index:
                continue
            numerator = numerator * (values - interpolation_nodes[other])
            denominator = denominator * (interpolation_nodes[index] - interpolation_nodes[other])
        columns.append(numerator / denominator)
    return jnp.stack(columns, axis=-1)


def growth_factor(redshift: Array, linear_theory: LinearTheoryTable) -> Array:
    """Interpolate the dimensionless PINOCCHIO linear growth factor."""

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    return _linear_interpolate(
        scale_factor,
        linear_theory.scale_factor,
        linear_theory.growth,
    )


def comoving_distance_mpc_h(
    redshift: Array,
    linear_theory: LinearTheoryTable,
) -> Array:
    """Interpolate PINOCCHIO comoving distance in ``Mpc/h``."""

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    return _linear_interpolate(
        scale_factor,
        linear_theory.scale_factor,
        linear_theory.chi_mpc_h,
    )


def redshift_at_comoving_distance(
    chi_mpc_h: Array,
    linear_theory: LinearTheoryTable,
) -> Array:
    """Interpolate redshift from comoving distance in ``Mpc/h``."""

    scale_factor = _linear_interpolate(
        jnp.asarray(chi_mpc_h),
        linear_theory.chi_mpc_h[::-1],
        linear_theory.scale_factor[::-1],
    )
    return 1.0 / scale_factor - 1.0


def linear_matter_power(
    k_h_mpc: Array,
    redshift: Array,
    linear_theory: LinearTheoryTable,
    power_evolution: LinearPowerEvolutionTable | None = None,
) -> Array:
    """Return linear matter power in ``(Mpc/h)^3``.

    The caller must validate that requested wavenumbers and redshifts lie
    within the PINOCCHIO tables. Values are clipped at table boundaries inside
    this JAX-compatible kernel to avoid dynamic host-side exceptions. When a
    scale-dependent table is supplied, interpolation is bilinear in scale
    factor and log wavenumber for log power. Otherwise the PINOCCHIO ``z=0``
    spectrum is scaled by its tabulated scalar growth factor squared.
    """

    k = jnp.asarray(k_h_mpc)
    if power_evolution is not None:
        scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
        k, scale_factor = jnp.broadcast_arrays(k, scale_factor)
        log_k_grid = jnp.log(power_evolution.k_h_mpc)
        log_k = jnp.clip(jnp.log(k), log_k_grid[0], log_k_grid[-1])
        scale_factor = jnp.clip(
            scale_factor,
            power_evolution.scale_factor[0],
            power_evolution.scale_factor[-1],
        )

        k_upper = jnp.clip(
            jnp.searchsorted(log_k_grid, log_k, side="right"),
            1,
            log_k_grid.shape[0] - 1,
        )
        k_lower = k_upper - 1
        scale_upper = jnp.clip(
            jnp.searchsorted(
                power_evolution.scale_factor,
                scale_factor,
                side="right",
            ),
            1,
            power_evolution.scale_factor.shape[0] - 1,
        )
        scale_lower = scale_upper - 1
        k_fraction = (log_k - log_k_grid[k_lower]) / (log_k_grid[k_upper] - log_k_grid[k_lower])
        scale_fraction = (scale_factor - power_evolution.scale_factor[scale_lower]) / (
            power_evolution.scale_factor[scale_upper] - power_evolution.scale_factor[scale_lower]
        )
        log_power = jnp.log(power_evolution.power_mpc_h3)
        lower_scale_power = (
            log_power[scale_lower, k_lower] * (1.0 - k_fraction)
            + log_power[scale_lower, k_upper] * k_fraction
        )
        upper_scale_power = (
            log_power[scale_upper, k_lower] * (1.0 - k_fraction)
            + log_power[scale_upper, k_upper] * k_fraction
        )
        return jnp.exp(
            lower_scale_power * (1.0 - scale_fraction) + upper_scale_power * scale_fraction
        )

    log_power = _linear_interpolate(
        jnp.log(k),
        jnp.log(linear_theory.k_h_mpc),
        jnp.log(linear_theory.power_mpc_h3),
    )
    return jnp.exp(log_power) * growth_factor(redshift, linear_theory) ** 2


def spherical_top_hat_window(x: Array) -> Array:
    """Return the Fourier-space spherical top-hat window.

    ``x`` is dimensionless. The small-argument series avoids cancellation and
    keeps the function finite and differentiable at the origin.
    """

    value = jnp.asarray(x)
    value_squared = value**2
    series = 1.0 - value_squared / 10.0 + value_squared**2 / 280.0
    safe_value = jnp.where(value == 0.0, 1.0, value)
    direct = 3.0 * (jnp.sin(value) - value * jnp.cos(value)) / safe_value**3
    # The direct numerator subtracts two nearly equal float32 values well
    # beyond machine precision for x << 0.1. The fourth-order series is still
    # accurate to better than 1e-10 at this switch.
    return jnp.where(jnp.abs(value) < 0.1, series, direct)


def linear_sigma_r(radius_mpc_h: Array, linear_theory: LinearTheoryTable) -> Array:
    """Return present-day linear ``sigma(R)`` for ``R`` in comoving ``Mpc/h``.

    The input PINOCCHIO spectrum is in ``(Mpc/h)^3`` at ``z=0`` and its
    wavenumbers are in ``h/Mpc``. Integration is over the complete tabulated
    logarithmic wavenumber range, without extrapolation.
    """

    radius = jnp.asarray(radius_mpc_h)
    k = linear_theory.k_h_mpc
    window = spherical_top_hat_window(k * radius[..., None])
    integrand = k**3 * linear_theory.power_mpc_h3 * window**2 / (2.0 * jnp.pi**2)
    variance = _trapezoid_last_axis(integrand, jnp.log(k))
    return jnp.sqrt(jnp.maximum(variance, 0.0))


def sigma8_from_linear_power(linear_theory: LinearTheoryTable) -> Array:
    """Return present-day linear ``sigma8`` reconstructed from PINOCCHIO ``P(k)``."""

    return linear_sigma_r(jnp.asarray(8.0), linear_theory)


def measured_hmf_dndlnm(
    redshift: Array,
    mass_function: HaloMassFunctionTable,
) -> Array:
    """Interpolate measured ``dn/dlnM`` in ``(Mpc/h)^-3`` at redshift."""

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    return _linear_interpolate(
        scale_factor,
        mass_function.scale_factor,
        mass_function.dndlnm_mpc_h3,
    )


def halo_count_weights(redshift: Array, quadrature: HaloCountQuadrature) -> Array:
    """Interpolate resolved number-density weights in ``(Mpc/h)^-3``.

    Returns shape ``redshift.shape + (n_mass,)``. Linear interpolation in
    scale factor preserves every representative-mass quadrature moment.
    Outside the input redshift range, the nearest snapshot is held fixed;
    callers requiring coverage must validate their requested interval.
    No interpolation or integration in mass is performed. Differentiable
    in the weights and in redshift away from interpolation knots.
    """

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    if quadrature.scale_factor.shape[0] == 1:
        return jnp.broadcast_to(
            quadrature.number_density_weight_mpc_h3[0],
            scale_factor.shape + (quadrature.log_mass_msun_h.shape[0],),
        )
    return _linear_interpolate(
        scale_factor,
        quadrature.scale_factor,
        quadrature.number_density_weight_mpc_h3,
    )


def _trapezoid_last_axis(values: Array, coordinate: Array) -> Array:
    widths = coordinate[1:] - coordinate[:-1]
    return jnp.sum(0.5 * (values[..., 1:] + values[..., :-1]) * widths, axis=-1)


def resolved_halo_mass_fraction(
    redshift: Array,
    mass_function: HaloMassFunctionTable,
    cosmology: Cosmology,
) -> Array:
    """Return the measured-HMF mass fraction represented by painted halos."""

    mass = jnp.exp(mass_function.log_mass_msun_h)
    dndlnm = measured_hmf_dndlnm(redshift, mass_function)
    mass_density = _trapezoid_last_axis(
        dndlnm * mass,
        mass_function.log_mass_msun_h,
    )
    return mass_density / rho_mean_comoving(cosmology)


def nfw_fourier_profile(
    k_h_mpc: Array,
    mass_msun_h: Array,
    redshift: Array,
    cosmology: Cosmology,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    quadrature: QuadratureRule | None = None,
) -> Array:
    """Return the mass-normalized Fourier transform of truncated 3D NFW halos.

    ``k_h_mpc`` is scalar or one-dimensional and ``mass_msun_h`` is
    one-dimensional. The result has shape ``(n_k, n_mass)`` or ``(n_mass,)``
    for scalar ``k``. A fixed radial quadrature keeps the transform
    differentiable with respect to all concentration parameters. Dividing by
    the quadrature's own zero-wavenumber integral enforces ``u(0)=1`` exactly.

    The tiny projected-map central softening is intentionally omitted from the
    continuum theory profile. Its effect is tested to remain below resolved
    map scales.
    """

    if quadrature is None:
        quadrature = gauss_legendre_rule(64)

    k_input = jnp.asarray(k_h_mpc)
    scalar_k = k_input.ndim == 0
    k_values = jnp.atleast_1d(k_input)
    mass = jnp.asarray(mass_msun_h)
    z = jnp.asarray(redshift)
    concentration = concentration_power_law(mass, z, concentration_params)
    overdensity = nfw_halo_overdensity(z, cosmology, profile_params)
    r_delta = halo_radius_delta_comoving(
        mass,
        z,
        cosmology,
        overdensity=overdensity,
        reference_density=profile_params.reference_density,
    )

    radius_fraction = 0.5 * (quadrature.nodes + 1.0)
    integration_weight = 0.5 * quadrature.weights
    nfw_mass_weight = (
        concentration[:, None] ** 2
        * radius_fraction[None, :]
        / (1.0 + concentration[:, None] * radius_fraction[None, :]) ** 2
        * integration_weight[None, :]
    )
    normalization = jnp.sum(nfw_mass_weight, axis=-1)

    def transform_one_k(k_value: Array) -> Array:
        phase = k_value * r_delta[:, None] * radius_fraction[None, :]
        sinc = jnp.sinc(phase / jnp.pi)
        transformed = jnp.sum(nfw_mass_weight * sinc, axis=-1) / normalization
        return jnp.where(k_value == 0.0, jnp.ones_like(transformed), transformed)

    transformed = lax.map(transform_one_k, k_values)
    return transformed[0] if scalar_k else transformed


def angular_support_radius(
    mass_msun_h: Array,
    redshift: Array,
    chi_mpc_h: Array,
    cosmology: Cosmology,
    profile_params: NFWProfileParams,
) -> Array:
    """Return the concentration-independent NFW support angle in radians."""

    overdensity = nfw_halo_overdensity(redshift, cosmology, profile_params)
    r_delta = halo_radius_delta_comoving(
        mass_msun_h,
        redshift,
        cosmology,
        overdensity=overdensity,
        reference_density=profile_params.reference_density,
    )
    argument = jnp.minimum(1.0, r_delta / (2.0 * chi_mpc_h))
    return 2.0 * jnp.arcsin(argument)


def halo_bias_at_redshift(redshift: Array, halo_bias: HaloBiasTable) -> Array:
    """Interpolate dimensionless corrected linear halo bias at redshift."""

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    return _linear_interpolate(
        scale_factor,
        halo_bias.scale_factor,
        halo_bias.linear_bias,
    )


def compensated_two_halo_response(
    k_h_mpc: Array,
    redshift: Array,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    halo_bias: HaloBiasTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    theta_resolution_rad: float | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> Array:
    """Return the compensated deterministic matter response.

    The response is

    ``1 + integral dlnM (dn/dlnM) (M/rho_mean) b(M,z) [u_NFW-W_L]``.

    Masses are ``Msun/h`` and wavenumbers are ``h/Mpc``. The measured HMF
    support is used without low-mass completion or bias renormalization.
    Since both Fourier profiles equal one at zero wavenumber, the response is
    exactly one there. The function is differentiable with respect to all
    concentration parameters; the supplied bias table is independent of
    concentration.
    """

    if halo_bias.log_mass_msun_h.shape != mass_function.log_mass_msun_h.shape:
        raise ValueError("halo-bias and mass-function mass grids must have matching shapes")
    if halo_bias.linear_bias.shape != (
        halo_bias.scale_factor.shape[0],
        mass_function.log_mass_msun_h.shape[0],
    ):
        raise ValueError("halo-bias arrays must have shape (n_scale_factor, n_mass)")

    mass = jnp.exp(mass_function.log_mass_msun_h)
    dndlnm = measured_hmf_dndlnm(redshift, mass_function)
    bias = halo_bias_at_redshift(redshift, halo_bias)
    cosmology = Cosmology(omega_m=linear_theory.omega_m0, h=linear_theory.h)
    profile = nfw_fourier_profile(
        k_h_mpc,
        mass,
        redshift,
        cosmology,
        concentration_params,
        profile_params,
        quadrature=profile_quadrature,
    )
    scalar_k = jnp.asarray(k_h_mpc).ndim == 0
    if scalar_k:
        profile = profile[None, :]

    if theta_resolution_rad is not None:
        chi = comoving_distance_mpc_h(redshift, linear_theory)
        theta = angular_support_radius(mass, redshift, chi, cosmology, profile_params)
        profile = jnp.where(theta[None, :] < theta_resolution_rad, 1.0, profile)

    mean_density = rho_mean_comoving(cosmology)
    lagrangian_radius = (3.0 * mass / (4.0 * jnp.pi * mean_density)) ** (1.0 / 3.0)
    k_values = jnp.atleast_1d(jnp.asarray(k_h_mpc))
    lagrangian_window = spherical_top_hat_window(
        k_values[:, None] * lagrangian_radius[None, :]
    )
    integrand = (
        dndlnm[None, :]
        * (mass[None, :] / mean_density)
        * bias[None, :]
        * (profile - lagrangian_window)
    )
    response = 1.0 + _trapezoid_last_axis(
        integrand,
        mass_function.log_mass_msun_h,
    )
    return response[0] if scalar_k else response


def two_halo_matter_power(
    k_h_mpc: Array,
    redshift: Array,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    halo_bias: HaloBiasTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    theta_resolution_rad: float | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> Array:
    """Return Castro-corrected compensated two-halo power in ``(Mpc/h)^3``."""

    response = compensated_two_halo_response(
        k_h_mpc,
        redshift,
        linear_theory,
        mass_function,
        halo_bias,
        concentration_params,
        profile_params,
        theta_resolution_rad=theta_resolution_rad,
        profile_quadrature=profile_quadrature,
    )
    return linear_matter_power(
        k_h_mpc,
        redshift,
        linear_theory,
        power_evolution,
    ) * response**2


def tabulate_shell_two_halo_response(
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    halo_bias: HaloBiasTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    k_h_mpc: Array | None = None,
    theta_resolution_rad: float | None = None,
    temporal_order: int = LINEAR_EVOLUTION_INTERPOLATION_ORDER,
    profile_quadrature: QuadratureRule | None = None,
) -> TwoHaloResponseTable:
    """Tabulate one shell's response for repeated angular projections."""

    temporal_quadrature = gauss_legendre_rule(temporal_order)
    _, redshift, _ = _shell_radial_quadrature(
        z_lo,
        z_hi,
        linear_theory,
        temporal_quadrature,
    )
    k_values = (
        linear_theory.k_h_mpc if k_h_mpc is None else jnp.asarray(k_h_mpc)
    )

    def response_at_redshift(redshift_value: Array) -> Array:
        return compensated_two_halo_response(
            k_values,
            redshift_value,
            linear_theory,
            mass_function,
            halo_bias,
            concentration_params,
            profile_params,
            theta_resolution_rad=theta_resolution_rad,
            profile_quadrature=profile_quadrature,
        )

    response = lax.map(response_at_redshift, redshift)
    scale_factor = 1.0 / (1.0 + redshift)
    return TwoHaloResponseTable(
        scale_factor=scale_factor[::-1],
        k_h_mpc=k_values,
        response=response[::-1],
    )


def interpolate_two_halo_response(
    k_h_mpc: Array,
    redshift: Array,
    response_table: TwoHaloResponseTable,
) -> Array:
    """Bilinearly interpolate a shell response in scale factor and log k."""

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    response_at_scale = _linear_interpolate(
        scale_factor,
        response_table.scale_factor,
        response_table.response,
    )
    return _linear_interpolate_last_axis(
        jnp.log(jnp.asarray(k_h_mpc)),
        jnp.log(response_table.k_h_mpc),
        response_at_scale,
    )


def one_halo_matter_power(
    k_h_mpc: Array,
    redshift: Array,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    theta_resolution_rad: float | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> Array:
    """Return compensated measured-HMF one-halo power in ``(Mpc/h)^3``.

    When ``theta_resolution_rad`` is supplied, halos below the map's angular
    NGP threshold use ``u=1``. The branch depends only on mass, redshift,
    distance, and profile support, never concentration. Supersampled and native
    resolved halos share the continuum NFW transform.

    Each halo is treated as a mass rearrangement from its mean-density
    Lagrangian top-hat patch, so the stochastic contribution is proportional
    to ``[u_NFW(k|M) - W_TH(k R_L)]^2``. Here
    ``R_L = [3 M / (4 pi rho_mean)]^(1/3)`` is in comoving ``Mpc/h``. This
    parameter-free compensation enforces the mass- and momentum-conserving
    ``P_1h = O(k^4)`` limit instead of adding an unphysical white-noise floor
    to the PINOCCHIO large-scale map.
    """

    mass = jnp.exp(mass_function.log_mass_msun_h)
    dndlnm = measured_hmf_dndlnm(redshift, mass_function)
    cosmology = Cosmology(
        omega_m=linear_theory.omega_m0,
        h=linear_theory.h,
    )
    profile = nfw_fourier_profile(
        k_h_mpc,
        mass,
        redshift,
        cosmology,
        concentration_params,
        profile_params,
        quadrature=profile_quadrature,
    )
    scalar_k = jnp.asarray(k_h_mpc).ndim == 0
    if scalar_k:
        profile = profile[None, :]

    if theta_resolution_rad is not None:
        chi = comoving_distance_mpc_h(redshift, linear_theory)
        theta = angular_support_radius(mass, redshift, chi, cosmology, profile_params)
        profile = jnp.where(theta[None, :] < theta_resolution_rad, 1.0, profile)

    mean_density = rho_mean_comoving(cosmology)
    lagrangian_radius = (3.0 * mass / (4.0 * jnp.pi * mean_density)) ** (1.0 / 3.0)
    k_values = jnp.atleast_1d(jnp.asarray(k_h_mpc))
    lagrangian_window = spherical_top_hat_window(k_values[:, None] * lagrangian_radius[None, :])
    compensated_profile = profile - lagrangian_window
    integrand = dndlnm[None, :] * (mass[None, :] / mean_density) ** 2 * compensated_profile**2
    result = _trapezoid_last_axis(integrand, mass_function.log_mass_msun_h)
    return result[0] if scalar_k else result


def normalized_halo_matter_power(
    k_h_mpc: Array,
    redshift: Array,
    linear_theory: LinearTheoryTable,
    halos: NormalizedHaloTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    theta_resolution_rad: float | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> NormalizedHaloPower:
    """Integrate the standard halo model using normalized fitted abundances.

    Returns ``I = integral f b u`` and the standard and compensated one-halo
    powers. The two-halo power is ``I**2 * P_linear(k,z)``, with no additive
    linear baseline or Lagrangian subtraction in I. All profile evaluations
    are shared and differentiable with respect to concentration parameters.
    Fitting and bias normalization are fixed host inputs. The optional NGP
    threshold is in radians, concentration independent, and sets ``u=1``.
    The analytic low-mass completion contributes to I; its omitted one-halo
    power must be bounded and convergence-tested by the caller.
    Wavenumbers are in h/Mpc and halo masses in Msun/h. I is dimensionless;
    both one-halo powers are in (comoving Mpc/h)^3.
    """

    a = 1.0 / (1.0 + jnp.asarray(redshift))
    weight = _linear_interpolate(a, halos.scale_factor, halos.mass_fraction_weight)
    biased_weight = _linear_interpolate(
        a, halos.scale_factor, halos.biased_mass_fraction_weight
    )
    tail = _linear_interpolate(a, halos.scale_factor, halos.tail_biased_mass_fraction)
    mass = halos.mass_msun_h
    cosmology = Cosmology(omega_m=linear_theory.omega_m0, h=linear_theory.h)
    profile = nfw_fourier_profile(
        k_h_mpc, mass, redshift, cosmology, concentration_params, profile_params,
        quadrature=profile_quadrature,
    )
    if theta_resolution_rad is not None:
        chi = comoving_distance_mpc_h(redshift, linear_theory)
        theta = angular_support_radius(mass, redshift, chi, cosmology, profile_params)
        profile = jnp.where(theta < theta_resolution_rad, 1.0, profile)
    mass_volume = mass / rho_mean_comoving(cosmology)
    lagrangian_radius = (3.0 * mass_volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    window = spherical_top_hat_window(jnp.asarray(k_h_mpc)[..., None] * lagrangian_radius)
    return NormalizedHaloPower(
        response=jnp.sum(biased_weight * profile, axis=-1) + tail,
        one_halo_standard=jnp.sum(weight * mass_volume * profile**2, axis=-1),
        one_halo_compensated=jnp.sum(weight * mass_volume * (profile - window)**2, axis=-1),
    )


def _shell_radial_quadrature(
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    quadrature: QuadratureRule,
) -> tuple[Array, Array, Array]:
    chi_lo = comoving_distance_mpc_h(jnp.asarray(z_lo), linear_theory)
    chi_hi = comoving_distance_mpc_h(jnp.asarray(z_hi), linear_theory)
    midpoint = 0.5 * (chi_lo + chi_hi)
    half_width = 0.5 * (chi_hi - chi_lo)
    chi = midpoint + half_width * quadrature.nodes
    dchi_weight = half_width * quadrature.weights
    redshift = redshift_at_comoving_distance(chi, linear_theory)
    return chi, redshift, dchi_weight


@partial(jit, static_argnames=("profile_params",))
def normalized_halo_power_grid(
    k_h_mpc: Array,
    redshift: Array,
    linear_theory: LinearTheoryTable,
    halos: NormalizedHaloTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams,
    *,
    theta_resolution_rad: float | None,
    profile_quadrature: QuadratureRule,
) -> NormalizedHaloPower:
    """Shared mass integrals with bounded k/redshift loops (shape n_z,n_k).

    Units and differentiability follow :func:`normalized_halo_matter_power`.
    Compilation is reused across shells with identical quadrature shapes.
    """

    return lax.map(lambda z: normalized_halo_matter_power(
        k_h_mpc, z, linear_theory, halos, concentration_params, profile_params,
        theta_resolution_rad=theta_resolution_rad, profile_quadrature=profile_quadrature,
    ), redshift)


def normalized_one_halo_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    power_table: NormalizedHaloPowerTable,
    *,
    radial_quadrature: QuadratureRule,
) -> tuple[Array, Array]:
    """Project both tabulated one-halo terms with the count-shell window.

    Returns dimensionless full-sky C_ell before the pixel window. The supplied
    k grid must cover the relevant support. Like the linear projection, power
    outside the supplied k range is zero, not a constant endpoint extension.
    """

    chi, redshift, dchi = _shell_radial_quadrature(z_lo, z_hi, linear_theory, radial_quadrature)
    lo = comoving_distance_mpc_h(jnp.asarray(z_lo), linear_theory)
    hi = comoving_distance_mpc_h(jnp.asarray(z_hi), linear_theory)
    prefactor = dchi * chi**2 / ((hi**3 - lo**3) / 3)**2
    k = (jnp.asarray(ell)[None, :] + .5) / chi[:, None]
    scale = 1 / (1 + redshift)
    result = []
    for values in (power_table.one_halo_standard, power_table.one_halo_compensated):
        at_z = _linear_interpolate(scale, power_table.scale_factor, values)
        at_k = lax.map(lambda pair: jnp.interp(
            pair[0], jnp.log(power_table.k_h_mpc), pair[1]
        ), (jnp.log(k), at_z))
        at_k = jnp.where((k >= power_table.k_h_mpc[0]) & (k <= power_table.k_h_mpc[-1]), at_k, 0.)
        result.append(jnp.sum(prefactor[:, None] * at_k, axis=0))
    return result[0], result[1]


def _limber_shell_components(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    response_table: TwoHaloResponseTable | None = None,
    theta_resolution_rad: float | None = None,
    radial_quadrature: QuadratureRule | None = None,
    profile_quadrature: QuadratureRule | None = None,
    compute_one_halo: bool = True,
) -> tuple[Array, Array, Array]:
    """Return Limber linear, two-halo, and one-halo shell spectra.

    The shell field is the count overdensity with normalized radial window
    ``W(chi) = 3 chi^2 / (chi_hi^3 - chi_lo^3)``. The one-halo result is fully
    differentiable with respect to the concentration parameters.
    """

    if radial_quadrature is None:
        radial_quadrature = gauss_legendre_rule(64)
    ell_values = jnp.asarray(ell)
    chi, redshift, dchi_weight = _shell_radial_quadrature(
        z_lo,
        z_hi,
        linear_theory,
        radial_quadrature,
    )
    chi_lo = comoving_distance_mpc_h(jnp.asarray(z_lo), linear_theory)
    chi_hi = comoving_distance_mpc_h(jnp.asarray(z_hi), linear_theory)
    shell_volume_per_sr = (chi_hi**3 - chi_lo**3) / 3.0
    radial_prefactor = dchi_weight * chi**2 / shell_volume_per_sr**2

    def node_power(inputs: tuple[Array, Array]) -> tuple[Array, Array, Array]:
        chi_node, redshift_node = inputs
        k = (ell_values + 0.5) / chi_node
        linear = linear_matter_power(
            k,
            redshift_node,
            linear_theory,
            power_evolution,
        )
        power_k_grid = linear_theory.k_h_mpc if power_evolution is None else power_evolution.k_h_mpc
        linear = jnp.where(
            (k >= power_k_grid[0]) & (k <= power_k_grid[-1]),
            linear,
            0.0,
        )
        if response_table is None:
            two_halo = linear
        else:
            response = interpolate_two_halo_response(k, redshift_node, response_table)
            two_halo = linear * response**2
        one_halo = one_halo_matter_power(
            k,
            redshift_node,
            linear_theory,
            mass_function,
            concentration_params,
            profile_params,
            theta_resolution_rad=theta_resolution_rad,
            profile_quadrature=profile_quadrature,
        ) if compute_one_halo else jnp.zeros_like(linear)
        return linear, two_halo, one_halo

    linear_nodes, two_halo_nodes, one_halo_nodes = lax.map(node_power, (chi, redshift))
    return (
        jnp.sum(radial_prefactor[:, None] * linear_nodes, axis=0),
        jnp.sum(radial_prefactor[:, None] * two_halo_nodes, axis=0),
        jnp.sum(radial_prefactor[:, None] * one_halo_nodes, axis=0),
    )


def limber_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    theta_resolution_rad: float | None = None,
    radial_quadrature: QuadratureRule | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> tuple[Array, Array]:
    """Return the legacy Limber linear and compensated one-halo spectra."""

    linear, _, one_halo = _limber_shell_components(
        ell,
        z_lo,
        z_hi,
        linear_theory,
        mass_function,
        concentration_params,
        profile_params,
        power_evolution=power_evolution,
        theta_resolution_rad=theta_resolution_rad,
        radial_quadrature=radial_quadrature,
        profile_quadrature=profile_quadrature,
    )
    return linear, one_halo


def limber_halo_model_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    concentration_params: ConcentrationParams,
    response_table: TwoHaloResponseTable,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    theta_resolution_rad: float | None = None,
    radial_quadrature: QuadratureRule | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> tuple[Array, Array, Array]:
    """Return Limber linear, corrected two-halo, and one-halo spectra."""

    return _limber_shell_components(
        ell,
        z_lo,
        z_hi,
        linear_theory,
        mass_function,
        concentration_params,
        profile_params,
        power_evolution=power_evolution,
        response_table=response_table,
        theta_resolution_rad=theta_resolution_rad,
        radial_quadrature=radial_quadrature,
        profile_quadrature=profile_quadrature,
    )


def _finite_width_flat_sky_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    response_table: TwoHaloResponseTable | None = None,
    radial_quadrature: QuadratureRule | None = None,
    line_of_sight_quadrature: QuadratureRule | None = None,
    line_of_sight_tail_periods: float = 40.0,
) -> Array:
    """Return one finite-width flat-sky deterministic shell spectrum.

    The shell field is the dimensionless count overdensity with radial window
    ``W(chi) = 3 chi^2 / (chi_hi^3 - chi_lo^3)`` for comoving ``chi`` in
    ``Mpc/h``. Unlike standard Limber, this approximation retains the Fourier
    transform of the complete top-hat radial window and its growth evolution:

    ``C_ell = integral dk_parallel |W_sqrtP(k_parallel)|^2 / (pi chi_mid^2)``.

    For scale-dependent evolution or a two-halo response,
    ``sqrt(P(k,z)) * response(k,z)`` is interpolated across the narrow shell
    with eight fixed Lagrange nodes before the radial Fourier transform. This
    keeps the full tabulated k dependence without constructing an
    ``n_ell * n_los * n_radial`` array. The scalar-growth linear fallback
    retains its direct radial quadrature.

    It is intended as the high-ell continuation for geometrically thin
    shells. The orchestration layer validates it against the exact full-sky
    projection before selecting it. The calculation is JAX-compatible and
    differentiable with respect to concentration parameters when a response
    table is supplied.
    """

    if z_hi <= z_lo:
        raise ValueError("finite-width shell must satisfy z_hi > z_lo")
    if line_of_sight_tail_periods <= 0.0:
        raise ValueError("line-of-sight tail periods must be positive")
    if radial_quadrature is None:
        radial_quadrature = gauss_legendre_rule(256)
    if line_of_sight_quadrature is None:
        line_of_sight_quadrature = gauss_legendre_rule(512)

    ell_values = jnp.asarray(ell)
    scalar_ell = ell_values.ndim == 0
    ell_vector = jnp.atleast_1d(ell_values)
    chi_lo = comoving_distance_mpc_h(jnp.asarray(z_lo), linear_theory)
    chi_hi = comoving_distance_mpc_h(jnp.asarray(z_hi), linear_theory)
    midpoint = 0.5 * (chi_lo + chi_hi)
    half_width = 0.5 * (chi_hi - chi_lo)
    chi = midpoint + half_width * radial_quadrature.nodes
    dchi_weight = half_width * radial_quadrature.weights
    redshift = redshift_at_comoving_distance(chi, linear_theory)
    shell_volume_per_sr = (chi_hi**3 - chi_lo**3) / 3.0
    radial_weight = dchi_weight * chi**2 / shell_volume_per_sr

    u_max = line_of_sight_tail_periods * jnp.pi
    u = 0.5 * u_max * (line_of_sight_quadrature.nodes + 1.0)
    du_weight = 0.5 * u_max * line_of_sight_quadrature.weights
    phase = u[:, None] * radial_quadrature.nodes[None, :]
    k_parallel = u / half_width
    power_k_grid = linear_theory.k_h_mpc if power_evolution is None else power_evolution.k_h_mpc

    use_temporal_basis = power_evolution is not None or response_table is not None
    if not use_temporal_basis:
        transfer_weight = radial_weight * growth_factor(redshift, linear_theory)
        radial_window_power = (
            jnp.sum(jnp.cos(phase) * transfer_weight[None, :], axis=1) ** 2
            + jnp.sum(jnp.sin(phase) * transfer_weight[None, :], axis=1) ** 2
        )
    else:
        evolution_rule = gauss_legendre_rule(LINEAR_EVOLUTION_INTERPOLATION_ORDER)
        sample_chi = midpoint + half_width * evolution_rule.nodes
        sample_redshift = redshift_at_comoving_distance(sample_chi, linear_theory)
        temporal_basis = _lagrange_basis(
            radial_quadrature.nodes,
            evolution_rule.nodes,
        )
        weighted_basis = radial_weight[:, None] * temporal_basis
        radial_basis_real = jnp.cos(phase) @ weighted_basis
        radial_basis_imag = jnp.sin(phase) @ weighted_basis

    def project_one_ell(ell_value: Array) -> Array:
        k_transverse = (ell_value + 0.5) / midpoint
        k = jnp.sqrt(k_transverse**2 + k_parallel**2)
        valid_k = (k >= power_k_grid[0]) & (k <= power_k_grid[-1])
        if not use_temporal_basis:
            power = linear_matter_power(k, jnp.zeros_like(k), linear_theory)
            integrand = jnp.where(valid_k, power * radial_window_power, 0.0)
        else:
            if power_evolution is None:
                power = linear_matter_power(k, jnp.zeros_like(k), linear_theory)
                sample_amplitude = jnp.broadcast_to(
                    growth_factor(sample_redshift, linear_theory)[None, :],
                    (k.shape[0], sample_redshift.shape[0]),
                )
            else:
                sample_power = linear_matter_power(
                    k[:, None],
                    sample_redshift[None, :],
                    linear_theory,
                    power_evolution,
                )
                sample_amplitude = jnp.sqrt(sample_power)
            if response_table is not None:
                sample_response = lax.map(
                    lambda sample_z: interpolate_two_halo_response(
                        k,
                        sample_z,
                        response_table,
                    ),
                    sample_redshift,
                ).T
                sample_amplitude = sample_amplitude * sample_response
            transformed_real = jnp.sum(
                radial_basis_real * sample_amplitude,
                axis=1,
            )
            transformed_imag = jnp.sum(
                radial_basis_imag * sample_amplitude,
                axis=1,
            )
            transformed_power = transformed_real**2 + transformed_imag**2
            if power_evolution is None:
                transformed_power = power * transformed_power
            integrand = jnp.where(valid_k, transformed_power, 0.0)
        return jnp.sum(du_weight * integrand) / (jnp.pi * midpoint**2 * half_width)

    result = lax.map(project_one_ell, ell_vector)
    return result[0] if scalar_ell else result


def finite_width_flat_sky_linear_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    radial_quadrature: QuadratureRule | None = None,
    line_of_sight_quadrature: QuadratureRule | None = None,
    line_of_sight_tail_periods: float = 40.0,
) -> Array:
    """Return the finite-width flat-sky linear ``C_ell`` for one count shell."""

    return _finite_width_flat_sky_shell_cls(
        ell,
        z_lo,
        z_hi,
        linear_theory,
        power_evolution=power_evolution,
        radial_quadrature=radial_quadrature,
        line_of_sight_quadrature=line_of_sight_quadrature,
        line_of_sight_tail_periods=line_of_sight_tail_periods,
    )


def finite_width_flat_sky_two_halo_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    response_table: TwoHaloResponseTable,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    radial_quadrature: QuadratureRule | None = None,
    line_of_sight_quadrature: QuadratureRule | None = None,
    line_of_sight_tail_periods: float = 40.0,
) -> Array:
    """Return the finite-width corrected two-halo ``C_ell`` for one shell.

    The response table is dimensionless and the result is differentiable with
    respect to response values, including their concentration dependence.
    """

    return _finite_width_flat_sky_shell_cls(
        ell,
        z_lo,
        z_hi,
        linear_theory,
        power_evolution=power_evolution,
        response_table=response_table,
        radial_quadrature=radial_quadrature,
        line_of_sight_quadrature=line_of_sight_quadrature,
        line_of_sight_tail_periods=line_of_sight_tail_periods,
    )


class _ExactProjectionState(NamedTuple):
    log_k_table: np.ndarray
    log_power_table: np.ndarray
    chi_nodes: np.ndarray
    transfer_weight: np.ndarray
    temporal_basis_weight: np.ndarray
    log_power_evolution: np.ndarray
    evolution_scale_lower: np.ndarray
    evolution_scale_fraction: np.ndarray
    temporal_growth: np.ndarray
    response: np.ndarray
    scale_dependent: bool
    has_response: bool
    weights: np.ndarray
    shell_midpoint: np.ndarray
    shell_width: np.ndarray
    k_table_max: float
    log_k_min: float
    relative_tolerance: float
    radial_tail_periods: float


_EXACT_PROJECTION_STATE: _ExactProjectionState | None = None


def _evolution_amplitude_at_log_k(
    log_k: float,
    state: _ExactProjectionState,
    shell_index: int | None = None,
) -> np.ndarray:
    """Return ``sqrt(P(k,a) / P(k,a=1))`` at exact temporal nodes."""

    upper_k = int(np.searchsorted(state.log_k_table, log_k, side="right"))
    upper_k = int(np.clip(upper_k, 1, state.log_k_table.size - 1))
    lower_k = upper_k - 1
    k_fraction = (log_k - state.log_k_table[lower_k]) / (
        state.log_k_table[upper_k] - state.log_k_table[lower_k]
    )
    lower_scale = state.evolution_scale_lower
    scale_fraction = state.evolution_scale_fraction
    if shell_index is not None:
        lower_scale = lower_scale[shell_index]
        scale_fraction = scale_fraction[shell_index]
    upper_scale = lower_scale + 1
    lower_log_power = (
        state.log_power_evolution[lower_scale, lower_k] * (1.0 - k_fraction)
        + state.log_power_evolution[lower_scale, upper_k] * k_fraction
    )
    upper_log_power = (
        state.log_power_evolution[upper_scale, lower_k] * (1.0 - k_fraction)
        + state.log_power_evolution[upper_scale, upper_k] * k_fraction
    )
    sample_log_power = lower_log_power * (1.0 - scale_fraction) + upper_log_power * scale_fraction
    reference_log_power = float(np.interp(log_k, state.log_k_table, state.log_power_table))
    return np.exp(0.5 * (sample_log_power - reference_log_power))


def _response_at_log_k(
    log_k: float,
    state: _ExactProjectionState,
    shell_index: int | None = None,
) -> np.ndarray:
    """Interpolate the deterministic two-halo response at temporal nodes."""

    upper_k = int(np.searchsorted(state.log_k_table, log_k, side="right"))
    upper_k = int(np.clip(upper_k, 1, state.log_k_table.size - 1))
    lower_k = upper_k - 1
    fraction = (log_k - state.log_k_table[lower_k]) / (
        state.log_k_table[upper_k] - state.log_k_table[lower_k]
    )
    response = state.response if shell_index is None else state.response[shell_index]
    return response[..., lower_k] * (1.0 - fraction) + response[..., upper_k] * fraction


def _integrate_exact_multipole(
    ell_value: int,
    state: _ExactProjectionState,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Integrate one exact multipole using process-local read-only state."""

    from scipy.integrate import quad, quad_vec
    from scipy.special import spherical_jn

    if ell_value < 0:
        raise ValueError("ell values must be non-negative")
    transverse_k = (ell_value + 0.5) / state.shell_midpoint
    transverse_periods = transverse_k * state.shell_width / np.pi
    tail_periods = np.clip(transverse_periods, 40.0, state.radial_tail_periods)
    radial_tail_k = tail_periods * np.pi / state.shell_width
    shell_k_max = np.minimum(state.k_table_max, transverse_k + radial_tail_k)
    k_max = float(np.max(shell_k_max))

    def power_at_log_k(log_k: float) -> tuple[float, float]:
        k = np.exp(log_k)
        power = np.exp(np.interp(log_k, state.log_k_table, state.log_power_table))
        return k, (2.0 / np.pi) * k**3 * power

    def shell_transfers(log_k: float, shell_index: int) -> tuple[float, float]:
        k, prefactor = power_at_log_k(log_k)
        bessel = spherical_jn(ell_value, k * state.chi_nodes[shell_index])
        if state.scale_dependent or state.has_response:
            basis_transfer = np.sum(
                state.temporal_basis_weight[shell_index] * bessel[:, None],
                axis=0,
            )
            if state.scale_dependent:
                evolution_amplitude = _evolution_amplitude_at_log_k(
                    log_k,
                    state,
                    shell_index,
                )
                linear_transfer = np.sum(basis_transfer * evolution_amplitude)
            else:
                evolution_amplitude = state.temporal_growth[shell_index]
                linear_transfer = np.sum(state.transfer_weight[shell_index] * bessel)
            if state.has_response:
                response = _response_at_log_k(log_k, state, shell_index)
                two_halo_transfer = np.sum(
                    basis_transfer * evolution_amplitude * response
                )
            else:
                two_halo_transfer = linear_transfer
        else:
            linear_transfer = np.sum(state.transfer_weight[shell_index] * bessel)
            two_halo_transfer = linear_transfer
        return (
            float(prefactor * linear_transfer**2),
            float(prefactor * two_halo_transfer**2),
        )

    shell_integrated = np.empty(state.chi_nodes.shape[0], dtype=np.float64)
    shell_two_halo_integrated = np.empty(state.chi_nodes.shape[0], dtype=np.float64)
    for shell_index, shell_limit in enumerate(shell_k_max):
        if state.has_response:
            integrated, _ = quad_vec(
                lambda log_k, index=shell_index: np.asarray(shell_transfers(log_k, index)),
                state.log_k_min,
                np.log(shell_limit),
                epsabs=1.0e-14,
                epsrel=state.relative_tolerance,
                limit=2000,
            )
            shell_integrated[shell_index] = integrated[0]
            shell_two_halo_integrated[shell_index] = integrated[1]
        else:
            shell_integrated[shell_index], _ = quad(
                lambda log_k, index=shell_index: shell_transfers(log_k, index)[0],
                state.log_k_min,
                np.log(shell_limit),
                epsabs=1.0e-14,
                epsrel=state.relative_tolerance,
                limit=2000,
            )
            shell_two_halo_integrated[shell_index] = shell_integrated[shell_index]

    def summed_transfers(log_k: float) -> tuple[float, float]:
        k, prefactor = power_at_log_k(log_k)
        bessel = spherical_jn(ell_value, k * state.chi_nodes)
        if state.scale_dependent or state.has_response:
            basis_transfer = np.sum(
                state.temporal_basis_weight * bessel[:, :, None],
                axis=1,
            )
            if state.scale_dependent:
                evolution_amplitude = _evolution_amplitude_at_log_k(log_k, state)
                linear_transfer = np.sum(basis_transfer * evolution_amplitude, axis=1)
            else:
                evolution_amplitude = state.temporal_growth
                linear_transfer = np.sum(state.transfer_weight * bessel, axis=1)
            if state.has_response:
                response = _response_at_log_k(log_k, state)
                two_halo_transfer = np.sum(
                    basis_transfer * evolution_amplitude * response,
                    axis=1,
                )
            else:
                two_halo_transfer = linear_transfer
        else:
            linear_transfer = np.sum(state.transfer_weight * bessel, axis=1)
            two_halo_transfer = linear_transfer
        linear_transfer = np.where(k <= shell_k_max, linear_transfer, 0.0)
        two_halo_transfer = np.where(k <= shell_k_max, two_halo_transfer, 0.0)
        summed_linear_transfer = np.sum(state.weights * linear_transfer)
        summed_two_halo_transfer = np.sum(state.weights * two_halo_transfer)
        return (
            float(prefactor * summed_linear_transfer**2),
            float(prefactor * summed_two_halo_transfer**2),
        )

    if state.has_response:
        summed_pair, _ = quad_vec(
            lambda log_k: np.asarray(summed_transfers(log_k)),
            state.log_k_min,
            np.log(k_max),
            epsabs=1.0e-14,
            epsrel=state.relative_tolerance,
            limit=2000,
        )
        summed_integrated = float(summed_pair[0])
        summed_two_halo_integrated = float(summed_pair[1])
    else:
        summed_integrated, _ = quad(
            lambda log_k: summed_transfers(log_k)[0],
            state.log_k_min,
            np.log(k_max),
            epsabs=1.0e-14,
            epsrel=state.relative_tolerance,
            limit=2000,
        )
        summed_two_halo_integrated = summed_integrated
    return (
        shell_integrated,
        float(summed_integrated),
        shell_two_halo_integrated,
        float(summed_two_halo_integrated),
    )


def _initialize_exact_projection_worker(state: _ExactProjectionState) -> None:
    global _EXACT_PROJECTION_STATE
    _EXACT_PROJECTION_STATE = state


def _integrate_exact_multipole_worker(
    ell_value: int,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    if _EXACT_PROJECTION_STATE is None:
        raise RuntimeError("exact projection worker was not initialized")
    return _integrate_exact_multipole(ell_value, _EXACT_PROJECTION_STATE)


def _numpy_lagrange_basis(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Evaluate fixed-node Lagrange basis polynomials for SciPy orchestration."""

    values = np.asarray(x, dtype=np.float64)
    interpolation_nodes = np.asarray(nodes, dtype=np.float64)
    basis = np.ones((values.size, interpolation_nodes.size), dtype=np.float64)
    for index in range(interpolation_nodes.size):
        for other in range(interpolation_nodes.size):
            if other == index:
                continue
            basis[:, index] *= (values - interpolation_nodes[other]) / (
                interpolation_nodes[index] - interpolation_nodes[other]
            )
    return basis


def _exact_deterministic_shell_cls(
    ell: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    linear_theory: LinearTheoryTable,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    response_tables: Sequence[TwoHaloResponseTable] | None = None,
    shell_weights: np.ndarray | None = None,
    radial_order: int = 512,
    radial_tail_periods: float = 256.0,
    relative_tolerance: float = 1.0e-4,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact low-ell linear and two-halo shell and summed spectra.

    This host-side orchestration path uses SciPy's spherical Bessel functions
    and adaptive scalar quadrature. Supplied response tables may encode a
    concentration-dependent profile, but conversion to NumPy makes this exact
    path non-differentiable. Shell autos use
    shell-specific wavenumber cutoffs, while the weighted sum retains
    cross-shell correlations up to each shell's cutoff. The radial tail spans
    at least 40 oscillation periods and grows with transverse wavenumber up to
    ``radial_tail_periods``. Independent multipoles are evaluated in
    ``workers`` spawned processes. Install
    ``geppetto[theory]`` to use it. The differentiable one-halo, Limber, and
    finite-width kernels do not depend on SciPy. When ``power_evolution`` is supplied, the
    transfer uses ``sqrt(P(k,z) / P(k,0))`` rather than a scalar growth
    factor. Eight temporal interpolation nodes per shell preserve the smooth
    scale-dependent evolution without expanding the full radial-by-k table.
    """

    try:
        __import__("scipy.integrate")
        __import__("scipy.special")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "exact shell projection requires scipy; install geppetto[theory]"
        ) from exc

    ell_values = np.asarray(ell, dtype=np.int64)
    z_lo_values = np.asarray(z_lo, dtype=np.float64)
    z_hi_values = np.asarray(z_hi, dtype=np.float64)
    if z_lo_values.shape != z_hi_values.shape or z_lo_values.ndim != 1:
        raise ValueError("z_lo and z_hi must be one-dimensional arrays with matching shapes")
    if np.any(z_hi_values <= z_lo_values):
        raise ValueError("each shell must satisfy z_hi > z_lo")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    if radial_tail_periods < 40.0:
        raise ValueError("radial_tail_periods must be at least 40")
    if workers < 1:
        raise ValueError("exact projection workers must be positive")

    n_shell = z_lo_values.size
    if response_tables is not None and len(response_tables) != n_shell:
        raise ValueError("response_tables must contain one table per shell")
    if shell_weights is None:
        weights = np.ones(n_shell, dtype=np.float64) / max(n_shell, 1)
    else:
        weights = np.asarray(shell_weights, dtype=np.float64)
        if weights.shape != (n_shell,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("shell_weights must be non-negative with shape (n_shell,)")
        weights = weights / np.sum(weights)

    scale_factor = np.asarray(linear_theory.scale_factor, dtype=np.float64)
    chi_table = np.asarray(linear_theory.chi_mpc_h, dtype=np.float64)
    growth_table = np.asarray(linear_theory.growth, dtype=np.float64)
    if power_evolution is None:
        k_table = np.asarray(linear_theory.k_h_mpc, dtype=np.float64)
        power_table = np.asarray(linear_theory.power_mpc_h3, dtype=np.float64)
        evolution_scale_factor = np.empty(0, dtype=np.float64)
        log_power_evolution = np.empty((0, 0), dtype=np.float64)
    else:
        evolution_scale_factor = np.asarray(
            power_evolution.scale_factor,
            dtype=np.float64,
        )
        k_table = np.asarray(power_evolution.k_h_mpc, dtype=np.float64)
        power_evolution_values = np.asarray(
            power_evolution.power_mpc_h3,
            dtype=np.float64,
        )
        if (
            evolution_scale_factor.ndim != 1
            or evolution_scale_factor.size < 2
            or not np.all(np.isfinite(evolution_scale_factor))
            or np.any(np.diff(evolution_scale_factor) <= 0.0)
            or not np.isclose(evolution_scale_factor[-1], 1.0, rtol=0.0, atol=1.0e-8)
            or k_table.ndim != 1
            or k_table.size < 2
            or not np.all(np.isfinite(k_table))
            or np.any(k_table <= 0.0)
            or np.any(np.diff(k_table) <= 0.0)
            or power_evolution_values.shape != (evolution_scale_factor.size, k_table.size)
            or np.any(power_evolution_values <= 0.0)
            or not np.all(np.isfinite(power_evolution_values))
        ):
            raise ValueError(
                "power_evolution must contain positive finite P(k,a) on increasing "
                "scale-factor and wavenumber grids ending at a=1"
            )
        power_table = power_evolution_values[-1]
        log_power_evolution = np.log(power_evolution_values)
    log_k_table = np.log(k_table)
    log_power_table = np.log(power_table)
    nodes, node_weights = np.polynomial.legendre.leggauss(radial_order)

    def chi_at_z(redshift: np.ndarray) -> np.ndarray:
        return np.interp(1.0 / (1.0 + redshift), scale_factor, chi_table)

    chi_lo = chi_at_z(z_lo_values)
    chi_hi = chi_at_z(z_hi_values)
    midpoint = 0.5 * (chi_lo + chi_hi)
    half_width = 0.5 * (chi_hi - chi_lo)
    chi_nodes = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    radial_weights = half_width[:, None] * node_weights[None, :]
    scale_nodes = np.interp(chi_nodes, chi_table[::-1], scale_factor[::-1])
    growth_nodes = np.interp(scale_nodes, scale_factor, growth_table)
    shell_volume_per_sr = (chi_hi**3 - chi_lo**3) / 3.0
    window = chi_nodes**2 / shell_volume_per_sr[:, None]
    transfer_weight = radial_weights * window * growth_nodes
    use_temporal_basis = power_evolution is not None or response_tables is not None
    if not use_temporal_basis:
        temporal_basis_weight = np.empty((0, 0, 0), dtype=np.float64)
        evolution_scale_lower = np.empty((0, 0), dtype=np.int64)
        evolution_scale_fraction = np.empty((0, 0), dtype=np.float64)
        temporal_growth = np.empty((0, 0), dtype=np.float64)
        response_values = np.empty((0, 0, 0), dtype=np.float64)
    else:
        evolution_nodes, _ = np.polynomial.legendre.leggauss(LINEAR_EVOLUTION_INTERPOLATION_ORDER)
        temporal_basis = _numpy_lagrange_basis(nodes, evolution_nodes)
        temporal_basis_weight = (
            radial_weights[:, :, None] * window[:, :, None] * temporal_basis[None, :, :]
        )
        temporal_chi = midpoint[:, None] + half_width[:, None] * evolution_nodes
        temporal_scale = np.interp(
            temporal_chi,
            chi_table[::-1],
            scale_factor[::-1],
        )
        temporal_growth = np.interp(temporal_scale, scale_factor, growth_table)
        if power_evolution is None:
            evolution_scale_lower = np.empty((0, 0), dtype=np.int64)
            evolution_scale_fraction = np.empty((0, 0), dtype=np.float64)
        else:
            if (
                np.min(temporal_scale) < evolution_scale_factor[0]
                or np.max(temporal_scale) > evolution_scale_factor[-1]
            ):
                raise ValueError("exact shell redshifts exceed the scale-dependent power table")
            evolution_scale_upper = np.searchsorted(
                evolution_scale_factor,
                temporal_scale,
                side="right",
            )
            evolution_scale_upper = np.clip(
                evolution_scale_upper,
                1,
                evolution_scale_factor.size - 1,
            )
            evolution_scale_lower = evolution_scale_upper - 1
            evolution_scale_fraction = (
                temporal_scale - evolution_scale_factor[evolution_scale_lower]
            ) / (
                evolution_scale_factor[evolution_scale_upper]
                - evolution_scale_factor[evolution_scale_lower]
            )

        if response_tables is None:
            response_values = np.empty((0, 0, 0), dtype=np.float64)
        else:
            response_values = np.empty(
                (n_shell, LINEAR_EVOLUTION_INTERPOLATION_ORDER, k_table.size),
                dtype=np.float64,
            )
            for shell_index, response_table in enumerate(response_tables):
                response_scale = np.asarray(response_table.scale_factor, dtype=np.float64)
                response_k = np.asarray(response_table.k_h_mpc, dtype=np.float64)
                response = np.asarray(response_table.response, dtype=np.float64)
                target_scale = temporal_scale[shell_index]
                scale_tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(response_scale))))
                k_tolerance = 1.0e-10 * max(1.0, float(response_k[-1]))
                if (
                    response_scale.ndim != 1
                    or response_k.ndim != 1
                    or response.shape != (response_scale.size, response_k.size)
                    or response_scale.size < 2
                    or response_k.size < 2
                    or np.any(np.diff(response_scale) <= 0.0)
                    or np.any(np.diff(response_k) <= 0.0)
                    or np.any(response_k <= 0.0)
                    or not np.all(np.isfinite(response))
                    or np.min(target_scale) < response_scale[0] - scale_tolerance
                    or np.max(target_scale) > response_scale[-1] + scale_tolerance
                    or k_table[0] < response_k[0] - k_tolerance
                    or k_table[-1] > response_k[-1] + k_tolerance
                ):
                    raise ValueError(
                        "two-halo response table does not cover the exact shell grid"
                    )
                target_scale = np.clip(target_scale, response_scale[0], response_scale[-1])
                upper_scale = np.clip(
                    np.searchsorted(response_scale, target_scale, side="right"),
                    1,
                    response_scale.size - 1,
                )
                lower_scale = upper_scale - 1
                scale_fraction = (target_scale - response_scale[lower_scale]) / (
                    response_scale[upper_scale] - response_scale[lower_scale]
                )
                response_at_scale = (
                    response[lower_scale] * (1.0 - scale_fraction[:, None])
                    + response[upper_scale] * scale_fraction[:, None]
                )
                response_values[shell_index] = np.stack(
                    [
                        np.interp(
                            log_k_table,
                            np.log(response_k),
                            response_row,
                        )
                        for response_row in response_at_scale
                    ]
                )

    shell_result = np.empty((n_shell, ell_values.size), dtype=np.float64)
    summed_result = np.empty(ell_values.size, dtype=np.float64)
    shell_two_halo_result = np.empty((n_shell, ell_values.size), dtype=np.float64)
    summed_two_halo_result = np.empty(ell_values.size, dtype=np.float64)
    shell_midpoint = 0.5 * (chi_lo + chi_hi)
    shell_width = chi_hi - chi_lo
    state = _ExactProjectionState(
        log_k_table=log_k_table,
        log_power_table=log_power_table,
        chi_nodes=chi_nodes,
        transfer_weight=transfer_weight,
        temporal_basis_weight=temporal_basis_weight,
        log_power_evolution=log_power_evolution,
        evolution_scale_lower=evolution_scale_lower,
        evolution_scale_fraction=evolution_scale_fraction,
        temporal_growth=temporal_growth,
        response=response_values,
        scale_dependent=power_evolution is not None,
        has_response=response_tables is not None,
        weights=weights,
        shell_midpoint=shell_midpoint,
        shell_width=shell_width,
        k_table_max=float(k_table[-1]),
        log_k_min=float(log_k_table[0]),
        relative_tolerance=relative_tolerance,
        radial_tail_periods=float(radial_tail_periods),
    )

    if workers == 1 or ell_values.size == 1:
        integrated_multipoles = (
            _integrate_exact_multipole(int(ell_value), state) for ell_value in ell_values
        )
        for ell_index, integrated in enumerate(integrated_multipoles):
            shell_integrated, sum_integrated, shell_two_halo, sum_two_halo = integrated
            shell_result[:, ell_index] = shell_integrated
            summed_result[ell_index] = sum_integrated
            shell_two_halo_result[:, ell_index] = shell_two_halo
            summed_two_halo_result[ell_index] = sum_two_halo
    else:
        affinity_environment = ("OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND")
        previous_affinity = {name: os.environ.get(name) for name in affinity_environment}
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ.pop("OMP_PLACES", None)
        os.environ["OMP_PROC_BIND"] = "FALSE"
        try:
            with ProcessPoolExecutor(
                max_workers=min(workers, ell_values.size),
                mp_context=get_context("spawn"),
                initializer=_initialize_exact_projection_worker,
                initargs=(state,),
            ) as executor:
                integrated_multipoles = executor.map(
                    _integrate_exact_multipole_worker,
                    (int(ell_value) for ell_value in ell_values),
                    chunksize=1,
                )
                for ell_index, integrated in enumerate(integrated_multipoles):
                    shell_integrated, sum_integrated, shell_two_halo, sum_two_halo = integrated
                    shell_result[:, ell_index] = shell_integrated
                    summed_result[ell_index] = sum_integrated
                    shell_two_halo_result[:, ell_index] = shell_two_halo
                    summed_two_halo_result[ell_index] = sum_two_halo
        finally:
            for name, previous_value in previous_affinity.items():
                if previous_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous_value

    return shell_result, summed_result, shell_two_halo_result, summed_two_halo_result


def exact_linear_shell_cls(
    ell: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    linear_theory: LinearTheoryTable,
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    shell_weights: np.ndarray | None = None,
    radial_order: int = 512,
    radial_tail_periods: float = 256.0,
    relative_tolerance: float = 1.0e-4,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact low-ell linear shell autos and weighted-sum spectrum."""

    shell, summed, _, _ = _exact_deterministic_shell_cls(
        ell,
        z_lo,
        z_hi,
        linear_theory,
        power_evolution=power_evolution,
        shell_weights=shell_weights,
        radial_order=radial_order,
        radial_tail_periods=radial_tail_periods,
        relative_tolerance=relative_tolerance,
        workers=workers,
    )
    return shell, summed


def exact_halo_model_shell_cls(
    ell: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    linear_theory: LinearTheoryTable,
    response_tables: Sequence[TwoHaloResponseTable],
    *,
    power_evolution: LinearPowerEvolutionTable | None = None,
    shell_weights: np.ndarray | None = None,
    radial_order: int = 512,
    radial_tail_periods: float = 256.0,
    relative_tolerance: float = 1.0e-4,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact linear and corrected two-halo shell and summed spectra.

    The SciPy projection is host-side and therefore not differentiable. Build
    and project Limber or finite-width response tables for concentration
    derivatives.
    """

    return _exact_deterministic_shell_cls(
        ell,
        z_lo,
        z_hi,
        linear_theory,
        power_evolution=power_evolution,
        response_tables=response_tables,
        shell_weights=shell_weights,
        radial_order=radial_order,
        radial_tail_periods=radial_tail_periods,
        relative_tolerance=relative_tolerance,
        workers=workers,
    )


def particle_count_shot_noise(
    mean_uncollapsed_counts_per_pixel: Array,
    mean_total_counts_per_pixel: Array,
    pixel_area_sr: float,
) -> Array:
    """Return white particle shot noise for total count-overdensity maps."""

    uncollapsed = jnp.asarray(mean_uncollapsed_counts_per_pixel)
    total = jnp.asarray(mean_total_counts_per_pixel)
    return pixel_area_sr * uncollapsed / total**2


def _stable_match_start(
    relative_error: np.ndarray,
    relative_tolerance: float,
    minimum_multipoles: int,
) -> int | None:
    """Return the first match that remains valid through the tested range."""

    errors = np.asarray(relative_error, dtype=np.float64)
    if errors.ndim != 1 or errors.size < minimum_multipoles:
        return None
    suffix_maximum = np.maximum.accumulate(errors[::-1])[::-1]
    candidate = np.flatnonzero(
        (suffix_maximum <= relative_tolerance)
        & (np.arange(errors.size) <= errors.size - minimum_multipoles)
    )
    return None if candidate.size == 0 else int(candidate[0])


def select_limber_transition(
    ell: np.ndarray,
    exact_shell: np.ndarray,
    exact_sum: np.ndarray,
    limber_shell: np.ndarray,
    limber_sum: np.ndarray,
    *,
    relative_tolerance: float,
    consecutive_multipoles: int,
) -> tuple[int | None, np.ndarray, float]:
    """Select a stable exact-to-Limber transition.

    A candidate is accepted only when every shell and the count-weighted sum
    remain within tolerance through the complete tested exact range, with at
    least ``consecutive_multipoles`` available. This rejects transient
    matches.
    """

    ell_values = np.asarray(ell, dtype=np.int64)
    exact_shell_values = np.asarray(exact_shell, dtype=np.float64)
    exact_sum_values = np.asarray(exact_sum, dtype=np.float64)
    limber_shell_values = np.asarray(limber_shell, dtype=np.float64)
    limber_sum_values = np.asarray(limber_sum, dtype=np.float64)
    if ell_values.ndim != 1 or ell_values.size == 0:
        raise ValueError("transition ell values must be a non-empty vector")
    if np.any(np.diff(ell_values) != 1):
        raise ValueError("transition ell values must be contiguous")
    if exact_shell_values.shape != limber_shell_values.shape or exact_shell_values.shape[1:] != (
        ell_values.size,
    ):
        raise ValueError("exact and Limber shell spectra must have shape (n_shell, n_ell)")
    if exact_sum_values.shape != (ell_values.size,) or limber_sum_values.shape != (
        ell_values.size,
    ):
        raise ValueError("exact and Limber summed spectra must have shape (n_ell,)")
    if relative_tolerance <= 0.0:
        raise ValueError("Limber relative tolerance must be positive")
    if consecutive_multipoles < 1:
        raise ValueError("Limber match width must be positive")

    shell_scale = np.maximum(np.abs(exact_shell_values), np.abs(limber_shell_values))
    sum_scale = np.maximum(np.abs(exact_sum_values), np.abs(limber_sum_values))
    tiny = np.finfo(np.float64).tiny
    shell_error = np.abs(exact_shell_values - limber_shell_values) / np.maximum(shell_scale, tiny)
    sum_error = np.abs(exact_sum_values - limber_sum_values) / np.maximum(sum_scale, tiny)
    worst_error = np.maximum(np.max(shell_error, axis=0), sum_error)
    start = _stable_match_start(
        worst_error,
        relative_tolerance,
        consecutive_multipoles,
    )
    if start is not None:
        return (
            int(ell_values[start]),
            np.max(shell_error[:, start:], axis=1),
            float(np.max(sum_error[start:])),
        )
    return None, np.max(shell_error, axis=1), float(np.max(sum_error))


def select_independent_limber_transitions(
    ell: np.ndarray,
    exact_shell: np.ndarray,
    exact_sum: np.ndarray,
    limber_shell: np.ndarray,
    limber_sum: np.ndarray,
    *,
    relative_tolerance: float,
    consecutive_multipoles: int,
) -> tuple[np.ndarray, int | None, np.ndarray, float]:
    """Select stable exact-to-Limber transitions for each spectrum.

    A match must remain within tolerance through the complete tested exact
    range. The returned shell transition array uses ``-1`` for spectra that
    have not matched. Unmatched errors are maxima over the final available
    window, which diagnoses the failure near the exact-projection cap.
    """

    ell_values = np.asarray(ell, dtype=np.int64)
    exact_shell_values = np.asarray(exact_shell, dtype=np.float64)
    exact_sum_values = np.asarray(exact_sum, dtype=np.float64)
    limber_shell_values = np.asarray(limber_shell, dtype=np.float64)
    limber_sum_values = np.asarray(limber_sum, dtype=np.float64)
    if ell_values.ndim != 1 or ell_values.size == 0:
        raise ValueError("transition ell values must be a non-empty vector")
    if np.any(np.diff(ell_values) != 1):
        raise ValueError("transition ell values must be contiguous")
    if exact_shell_values.shape != limber_shell_values.shape or exact_shell_values.shape[1:] != (
        ell_values.size,
    ):
        raise ValueError("exact and Limber shell spectra must have shape (n_shell, n_ell)")
    if exact_sum_values.shape != (ell_values.size,) or limber_sum_values.shape != (
        ell_values.size,
    ):
        raise ValueError("exact and Limber summed spectra must have shape (n_ell,)")
    if relative_tolerance <= 0.0:
        raise ValueError("Limber relative tolerance must be positive")
    if consecutive_multipoles < 1:
        raise ValueError("Limber match width must be positive")

    tiny = np.finfo(np.float64).tiny
    shell_scale = np.maximum(np.abs(exact_shell_values), np.abs(limber_shell_values))
    shell_error_values = np.abs(exact_shell_values - limber_shell_values) / np.maximum(
        shell_scale, tiny
    )
    sum_scale = np.maximum(np.abs(exact_sum_values), np.abs(limber_sum_values))
    sum_error_values = np.abs(exact_sum_values - limber_sum_values) / np.maximum(sum_scale, tiny)
    final_start = max(0, ell_values.size - consecutive_multipoles)

    shell_transition = np.full(exact_shell_values.shape[0], -1, dtype=np.int64)
    shell_error = np.empty(exact_shell_values.shape[0], dtype=np.float64)
    for shell_index, errors in enumerate(shell_error_values):
        accepted_start = _stable_match_start(
            errors,
            relative_tolerance,
            consecutive_multipoles,
        )
        if accepted_start is None:
            shell_error[shell_index] = float(np.max(errors[final_start:]))
        else:
            shell_transition[shell_index] = int(ell_values[accepted_start])
            shell_error[shell_index] = float(np.max(errors[accepted_start:]))

    summed_start = _stable_match_start(
        sum_error_values,
        relative_tolerance,
        consecutive_multipoles,
    )
    if summed_start is None:
        summed_transition = None
        summed_error = float(np.max(sum_error_values[final_start:]))
    else:
        summed_transition = int(ell_values[summed_start])
        summed_error = float(np.max(sum_error_values[summed_start:]))
    return shell_transition, summed_transition, shell_error, summed_error


def select_shell_high_ell_projection(
    exact_shell: np.ndarray,
    candidate_shell: np.ndarray,
    candidate_modes: np.ndarray,
    *,
    comparison_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the most accurate high-ell candidate for each shell.

    ``candidate_shell`` has shape ``(n_candidate, n_shell, n_ell)`` over the
    exact comparison range. Selection minimizes the maximum relative error in
    the final ``comparison_width`` multipoles. The returned mode identifiers
    correspond to ``candidate_modes``.
    """

    exact_values = np.asarray(exact_shell, dtype=np.float64)
    candidate_values = np.asarray(candidate_shell, dtype=np.float64)
    modes = np.asarray(candidate_modes, dtype=np.int64)
    if exact_values.ndim != 2:
        raise ValueError("exact shell spectra must have shape (n_shell, n_ell)")
    if (
        candidate_values.ndim != 3
        or candidate_values.shape[1:] != exact_values.shape
        or modes.shape != (candidate_values.shape[0],)
    ):
        raise ValueError("candidate shell spectra must have shape (n_candidate, n_shell, n_ell)")
    if comparison_width < 1 or exact_values.shape[1] < comparison_width:
        raise ValueError("projection comparison width exceeds the exact range")
    if not np.all(np.isfinite(candidate_values)):
        raise ValueError("high-ell projection candidates must be finite")

    tiny = np.finfo(np.float64).tiny
    scale = np.maximum(np.abs(candidate_values), np.abs(exact_values)[None, :, :])
    relative_error = np.abs(candidate_values - exact_values[None, :, :]) / np.maximum(
        scale,
        tiny,
    )
    final_error = np.max(relative_error[:, :, -comparison_width:], axis=2)
    selected_candidate = np.argmin(final_error, axis=0)
    shell_index = np.arange(exact_values.shape[0])
    return (
        candidate_values[selected_candidate, shell_index],
        modes[selected_candidate],
    )


def hybrid_angular_power_spectra(
    ell: Array,
    z_lo: Sequence[float],
    z_hi: Sequence[float],
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    concentration_params: ConcentrationParams,
    profile_params: Sequence[NFWProfileParams],
    *,
    shell_weights: Array,
    halo_bias: HaloBiasTable | None = None,
    two_halo_response_tables: Sequence[TwoHaloResponseTable] | None = None,
    one_halo_shell_cls: Array | None = None,
    power_evolution: LinearPowerEvolutionTable | None = None,
    pixel_window: Array | None = None,
    mean_uncollapsed_counts_per_pixel: Array | None = None,
    mean_total_counts_per_pixel: Array | None = None,
    pixel_area_sr: float = 1.0,
    theta_resolution_rad: float | None = None,
    ell_exact_cap: int = 512,
    limber_match_rtol: float = 0.01,
    limber_match_width: int = 20,
    exact_batch_size: int = 64,
    exact_workers: int = 1,
    exact_batch_callback: Callable[[str, int, int], None] | None = None,
    exact_batch_evaluator: Callable[
        [np.ndarray],
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]
    | None = None,
    radial_order: int = 64,
    exact_radial_order: int = 512,
    exact_radial_tail_periods: float = 256.0,
    finite_width_radial_order: int = 256,
    finite_width_line_of_sight_order: int = 512,
    finite_width_tail_periods: float = 40.0,
    profile_order: int = 64,
    exact_relative_tolerance: float = 1.0e-4,
) -> AngularPowerSpectra:
    """Compute hybrid exact/high-ell spectra for disjoint PINOCCHIO shells.

    Shells select between standard Limber and a finite-width flat-sky
    projection by comparison with the exact spherical-Bessel result. The
    count-weighted summed spectrum retains standard Limber because its broad
    radial window includes cross-shell correlations in the exact branch.
    Each transition must remain within ``limber_match_rtol`` through the
    complete exact range. The search is bounded by ``ell_exact_cap``. A zero
    cap uses a geometric shell-mode fallback without exact validation.
    One-halo power always uses Limber. When ``halo_bias`` or precomputed
    ``two_halo_response_tables`` are supplied, their two-halo spectrum replaces
    the linear spectrum in the clustering total; the baseline linear spectrum
    remains a diagnostic. ``halo_bias`` builds the legacy compensated response;
    supplied tables can instead carry the standard normalized-HMF response.
    Precomputed tables also avoid rebuilding responses used by an exact batch
    evaluator.
    ``one_halo_shell_cls`` can supply a precomputed, unwindowed one-halo
    spectrum of shape ``(n_shell, n_ell)``, avoiding duplicate profile work
    when the standard and compensated one-halo terms were computed together.
    ``power_evolution`` carries the
    optional scale-dependent PINOCCHIO CAMB series through every linear
    projection branch; omitting it retains scalar growth.
    """

    ell_values = jnp.asarray(ell)
    z_lo_values = np.asarray(z_lo, dtype=np.float64)
    z_hi_values = np.asarray(z_hi, dtype=np.float64)
    profiles = tuple(profile_params)
    if z_lo_values.shape != z_hi_values.shape or z_lo_values.ndim != 1:
        raise ValueError("z_lo and z_hi must be matching one-dimensional arrays")
    if len(profiles) != z_lo_values.size:
        raise ValueError("profile_params must contain one entry per shell")
    ell_numpy = np.asarray(ell_values, dtype=np.int64)
    if ell_numpy.ndim != 1 or ell_numpy.size == 0 or np.any(np.diff(ell_numpy) <= 0):
        raise ValueError("ell must be a non-empty vector of increasing multipoles")
    if ell_exact_cap < 0:
        raise ValueError("ell_exact_cap must be non-negative")
    if (
        limber_match_rtol <= 0.0
        or limber_match_width < 1
        or exact_batch_size < 1
        or exact_workers < 1
        or exact_radial_order < 2
        or exact_radial_tail_periods < 40.0
        or finite_width_radial_order < 2
        or finite_width_line_of_sight_order < 2
        or finite_width_tail_periods <= 0.0
    ):
        raise ValueError("Limber and exact projection controls must be positive")
    if ell_exact_cap > 0 and np.any(np.diff(ell_numpy) != 1):
        raise ValueError("adaptive exact-to-Limber selection requires contiguous multipoles")

    weights = jnp.asarray(shell_weights)
    if weights.shape != (z_lo_values.size,):
        raise ValueError("shell_weights must have shape (n_shell,)")
    weights = weights / jnp.sum(weights)
    radial_quadrature = gauss_legendre_rule(radial_order)
    finite_width_radial_quadrature = gauss_legendre_rule(finite_width_radial_order)
    finite_width_line_of_sight_quadrature = gauss_legendre_rule(finite_width_line_of_sight_order)
    profile_quadrature = gauss_legendre_rule(profile_order)
    response_k = linear_theory.k_h_mpc if power_evolution is None else power_evolution.k_h_mpc
    if two_halo_response_tables is not None:
        response_tables = tuple(two_halo_response_tables)
        if len(response_tables) != z_lo_values.size:
            raise ValueError("two_halo_response_tables must contain one table per shell")
    elif halo_bias is None:
        response_tables = None
    else:
        response_tables = tuple(
            tabulate_shell_two_halo_response(
                float(lo),
                float(hi),
                linear_theory,
                mass_function,
                halo_bias,
                concentration_params,
                shell_profile,
                k_h_mpc=response_k,
                theta_resolution_rad=theta_resolution_rad,
                profile_quadrature=profile_quadrature,
            )
            for lo, hi, shell_profile in zip(
                z_lo_values,
                z_hi_values,
                profiles,
                strict=True,
            )
        )
    if one_halo_shell_cls is not None:
        supplied_one_halo = jnp.asarray(one_halo_shell_cls)
        if supplied_one_halo.shape != (z_lo_values.size, ell_values.size):
            raise ValueError("one_halo_shell_cls must have shape (n_shell, n_ell)")
        limber_results = [
            _limber_shell_components(
                ell_values, float(lo), float(hi), linear_theory, mass_function,
                concentration_params, shell_profile, power_evolution=power_evolution,
                response_table=None if response_tables is None else response_tables[index],
                radial_quadrature=radial_quadrature, compute_one_halo=False,
            )[:2] + (supplied_one_halo[index],)
            for index, (lo, hi, shell_profile) in enumerate(zip(
                z_lo_values, z_hi_values, profiles, strict=True
            ))
        ]
    elif response_tables is None:
        legacy_limber_results = [
            limber_shell_cls(
                ell_values,
                float(lo),
                float(hi),
                linear_theory,
                mass_function,
                concentration_params,
                shell_profile,
                power_evolution=power_evolution,
                theta_resolution_rad=theta_resolution_rad,
                radial_quadrature=radial_quadrature,
                profile_quadrature=profile_quadrature,
            )
            for lo, hi, shell_profile in zip(
                z_lo_values,
                z_hi_values,
                profiles,
                strict=True,
            )
        ]
        limber_results = [
            (linear, linear, one_halo) for linear, one_halo in legacy_limber_results
        ]
    else:
        limber_results = [
            limber_halo_model_shell_cls(
                ell_values,
                float(lo),
                float(hi),
                linear_theory,
                mass_function,
                concentration_params,
                response_table,
                shell_profile,
                power_evolution=power_evolution,
                theta_resolution_rad=theta_resolution_rad,
                radial_quadrature=radial_quadrature,
                profile_quadrature=profile_quadrature,
            )
            for lo, hi, shell_profile, response_table in zip(
                z_lo_values,
                z_hi_values,
                profiles,
                response_tables,
                strict=True,
            )
        ]
    shell_linear = jnp.stack([result[0] for result in limber_results])
    shell_limber_linear = shell_linear
    shell_two_halo = jnp.stack([result[1] for result in limber_results])
    shell_limber_two_halo = shell_two_halo
    shell_finite_width_linear = jnp.stack(
        [
            finite_width_flat_sky_linear_shell_cls(
                ell_values,
                float(lo),
                float(hi),
                linear_theory,
                power_evolution=power_evolution,
                radial_quadrature=finite_width_radial_quadrature,
                line_of_sight_quadrature=finite_width_line_of_sight_quadrature,
                line_of_sight_tail_periods=finite_width_tail_periods,
            )
            for lo, hi in zip(z_lo_values, z_hi_values, strict=True)
        ]
    )
    shell_finite_width_two_halo = (
        shell_finite_width_linear
        if response_tables is None
        else jnp.stack(
            [
                finite_width_flat_sky_two_halo_shell_cls(
                    ell_values,
                    float(lo),
                    float(hi),
                    linear_theory,
                    response_table,
                    power_evolution=power_evolution,
                    radial_quadrature=finite_width_radial_quadrature,
                    line_of_sight_quadrature=finite_width_line_of_sight_quadrature,
                    line_of_sight_tail_periods=finite_width_tail_periods,
                )
                for lo, hi, response_table in zip(
                    z_lo_values,
                    z_hi_values,
                    response_tables,
                    strict=True,
                )
            ]
        )
    )
    shell_one_halo = jnp.stack([result[2] for result in limber_results])
    summed_linear = jnp.sum(weights[:, None] ** 2 * shell_linear, axis=0)
    summed_two_halo = jnp.sum(weights[:, None] ** 2 * shell_two_halo, axis=0)
    summed_one_halo = jnp.sum(weights[:, None] ** 2 * shell_one_halo, axis=0)

    shell_ell_high_ell_start = np.full(
        z_lo_values.size,
        int(ell_numpy[0]),
        dtype=np.int64,
    )
    summed_ell_limber_start = int(ell_numpy[0])
    shell_match_error = np.full(z_lo_values.size, np.nan, dtype=np.float64)
    summed_match_error = np.nan
    chi_lo_values = np.asarray(comoving_distance_mpc_h(jnp.asarray(z_lo_values), linear_theory))
    chi_hi_values = np.asarray(comoving_distance_mpc_h(jnp.asarray(z_hi_values), linear_theory))
    fractional_width = (chi_hi_values - chi_lo_values) / (0.5 * (chi_hi_values + chi_lo_values))
    shell_high_ell_mode = np.where(
        fractional_width <= 0.3,
        LINEAR_HIGH_ELL_FINITE_WIDTH,
        LINEAR_HIGH_ELL_LIMBER,
    )
    shell_linear = jnp.where(
        jnp.asarray(shell_high_ell_mode)[:, None] == LINEAR_HIGH_ELL_FINITE_WIDTH,
        shell_finite_width_linear,
        shell_limber_linear,
    )
    shell_two_halo = jnp.where(
        jnp.asarray(shell_high_ell_mode)[:, None] == LINEAR_HIGH_ELL_FINITE_WIDTH,
        shell_finite_width_two_halo,
        shell_limber_two_halo,
    )
    if ell_exact_cap > 0 and np.any(ell_numpy <= ell_exact_cap):
        exact_indices_all = np.flatnonzero(ell_numpy <= ell_exact_cap)
        exact_shell_blocks: list[np.ndarray] = []
        exact_sum_blocks: list[np.ndarray] = []
        exact_two_halo_shell_blocks: list[np.ndarray] = []
        exact_two_halo_sum_blocks: list[np.ndarray] = []
        for batch_start in range(0, exact_indices_all.size, exact_batch_size):
            batch_indices = exact_indices_all[batch_start : batch_start + exact_batch_size]
            batch_ell_min = int(ell_numpy[batch_indices[0]])
            batch_ell_max = int(ell_numpy[batch_indices[-1]])
            if exact_batch_callback is not None:
                exact_batch_callback("start", batch_ell_min, batch_ell_max)
            if exact_batch_evaluator is None:
                if response_tables is None:
                    exact_shell_batch, exact_sum_batch = exact_linear_shell_cls(
                        ell_numpy[batch_indices],
                        z_lo_values,
                        z_hi_values,
                        linear_theory,
                        power_evolution=power_evolution,
                        shell_weights=np.asarray(weights),
                        radial_order=exact_radial_order,
                        radial_tail_periods=exact_radial_tail_periods,
                        relative_tolerance=exact_relative_tolerance,
                        workers=exact_workers,
                    )
                    exact_two_halo_shell_batch = exact_shell_batch
                    exact_two_halo_sum_batch = exact_sum_batch
                else:
                    (
                        exact_shell_batch,
                        exact_sum_batch,
                        exact_two_halo_shell_batch,
                        exact_two_halo_sum_batch,
                    ) = exact_halo_model_shell_cls(
                        ell_numpy[batch_indices],
                        z_lo_values,
                        z_hi_values,
                        linear_theory,
                        response_tables,
                        power_evolution=power_evolution,
                        shell_weights=np.asarray(weights),
                        radial_order=exact_radial_order,
                        radial_tail_periods=exact_radial_tail_periods,
                        relative_tolerance=exact_relative_tolerance,
                        workers=exact_workers,
                    )
            else:
                exact_result = exact_batch_evaluator(ell_numpy[batch_indices])
                if len(exact_result) == 2 and response_tables is None:
                    exact_shell_batch, exact_sum_batch = exact_result
                    exact_two_halo_shell_batch = exact_shell_batch
                    exact_two_halo_sum_batch = exact_sum_batch
                elif len(exact_result) == 4:
                    (
                        exact_shell_batch,
                        exact_sum_batch,
                        exact_two_halo_shell_batch,
                        exact_two_halo_sum_batch,
                    ) = exact_result
                else:
                    raise ValueError(
                        "exact batch evaluator must return two linear arrays or four "
                        "linear and two-halo arrays"
                    )
            exact_shell_batch = np.asarray(exact_shell_batch, dtype=np.float64)
            exact_sum_batch = np.asarray(exact_sum_batch, dtype=np.float64)
            exact_two_halo_shell_batch = np.asarray(
                exact_two_halo_shell_batch,
                dtype=np.float64,
            )
            exact_two_halo_sum_batch = np.asarray(
                exact_two_halo_sum_batch,
                dtype=np.float64,
            )
            expected_shell_shape = (z_lo_values.size, batch_indices.size)
            if (
                exact_shell_batch.shape != expected_shell_shape
                or exact_two_halo_shell_batch.shape != expected_shell_shape
                or exact_sum_batch.shape != (batch_indices.size,)
                or exact_two_halo_sum_batch.shape != (batch_indices.size,)
            ):
                raise ValueError("exact batch evaluator returned inconsistent spectrum shapes")
            if not all(
                np.all(np.isfinite(values))
                for values in (
                    exact_shell_batch,
                    exact_sum_batch,
                    exact_two_halo_shell_batch,
                    exact_two_halo_sum_batch,
                )
            ):
                raise ValueError("exact batch evaluator returned non-finite spectra")
            if exact_batch_callback is not None:
                exact_batch_callback("complete", batch_ell_min, batch_ell_max)
            exact_shell_blocks.append(exact_shell_batch)
            exact_sum_blocks.append(exact_sum_batch)
            exact_two_halo_shell_blocks.append(exact_two_halo_shell_batch)
            exact_two_halo_sum_blocks.append(exact_two_halo_sum_batch)

        exact_shell = np.concatenate(exact_shell_blocks, axis=1)
        exact_sum = np.concatenate(exact_sum_blocks)
        exact_two_halo_shell = np.concatenate(exact_two_halo_shell_blocks, axis=1)
        exact_two_halo_sum = np.concatenate(exact_two_halo_sum_blocks)
        exact_indices = exact_indices_all
        if ell_numpy[-1] <= ell_exact_cap:
            shell_ell_high_ell_start.fill(int(ell_numpy[-1]) + 1)
            summed_ell_limber_start = int(ell_numpy[-1]) + 1
        else:
            selected_exact_shell, shell_high_ell_mode = select_shell_high_ell_projection(
                exact_two_halo_shell,
                np.stack(
                    (
                        np.asarray(shell_limber_two_halo)[:, exact_indices],
                        np.asarray(shell_finite_width_two_halo)[:, exact_indices],
                    )
                ),
                np.asarray(
                    (
                        LINEAR_HIGH_ELL_LIMBER,
                        LINEAR_HIGH_ELL_FINITE_WIDTH,
                    )
                ),
                comparison_width=limber_match_width,
            )
            shell_linear = jnp.where(
                jnp.asarray(shell_high_ell_mode)[:, None] == LINEAR_HIGH_ELL_FINITE_WIDTH,
                shell_finite_width_linear,
                shell_limber_linear,
            )
            shell_two_halo = jnp.where(
                jnp.asarray(shell_high_ell_mode)[:, None] == LINEAR_HIGH_ELL_FINITE_WIDTH,
                shell_finite_width_two_halo,
                shell_limber_two_halo,
            )
            (
                shell_transition,
                summed_transition,
                shell_match_error,
                summed_match_error,
            ) = select_independent_limber_transitions(
                ell_numpy[exact_indices],
                exact_two_halo_shell,
                exact_two_halo_sum,
                selected_exact_shell,
                np.asarray(summed_two_halo)[exact_indices],
                relative_tolerance=limber_match_rtol,
                consecutive_multipoles=limber_match_width,
            )
            if np.any(shell_transition < 0) or summed_transition is None:
                unmatched_shells = np.flatnonzero(shell_transition < 0)
                if unmatched_shells.size and (
                    summed_transition is not None
                    or np.max(shell_match_error[unmatched_shells]) >= summed_match_error
                ):
                    worst_shell = int(
                        unmatched_shells[np.argmax(shell_match_error[unmatched_shells])]
                    )
                    worst_label = f"shell={worst_shell}"
                    worst_error = float(shell_match_error[worst_shell])
                else:
                    worst_label = "summed_spectrum"
                    worst_error = float(summed_match_error)
                final_window_start = int(
                    ell_numpy[exact_indices[max(0, exact_indices.size - limber_match_width)]]
                )
                raise ValueError(
                    "an exact and high-ell two-halo projection did not converge before "
                    f"ell_exact_cap={ell_exact_cap}: {worst_label}, "
                    f"final_window={final_window_start}-{ell_numpy[exact_indices[-1]]}, "
                    f"maximum_relative_error={worst_error:.6g}"
                )
            shell_ell_high_ell_start = shell_transition
            summed_ell_limber_start = int(summed_transition)
        for shell_index, transition in enumerate(shell_ell_high_ell_start):
            use_exact = exact_indices[ell_numpy[exact_indices] < transition]
            if use_exact.size:
                exact_lookup = np.searchsorted(exact_indices, use_exact)
                shell_linear = shell_linear.at[shell_index, jnp.asarray(use_exact)].set(
                    jnp.asarray(exact_shell[shell_index, exact_lookup], dtype=shell_linear.dtype)
                )
                shell_two_halo = shell_two_halo.at[
                    shell_index,
                    jnp.asarray(use_exact),
                ].set(
                    jnp.asarray(
                        exact_two_halo_shell[shell_index, exact_lookup],
                        dtype=shell_two_halo.dtype,
                    )
                )
        use_exact_sum = exact_indices[ell_numpy[exact_indices] < summed_ell_limber_start]
        if use_exact_sum.size:
            exact_lookup = np.searchsorted(exact_indices, use_exact_sum)
            summed_linear = summed_linear.at[jnp.asarray(use_exact_sum)].set(
                jnp.asarray(exact_sum[exact_lookup], dtype=summed_linear.dtype)
            )
            summed_two_halo = summed_two_halo.at[jnp.asarray(use_exact_sum)].set(
                jnp.asarray(
                    exact_two_halo_sum[exact_lookup],
                    dtype=summed_two_halo.dtype,
                )
            )

    if pixel_window is None:
        pixel_window_values = jnp.ones_like(ell_values, dtype=shell_linear.dtype)
    else:
        pixel_window_values = jnp.asarray(pixel_window)
        if pixel_window_values.shape != ell_values.shape:
            raise ValueError("pixel_window must have shape (n_ell,)")
    pixel_window_squared = pixel_window_values**2
    shell_linear = shell_linear * pixel_window_squared[None, :]
    shell_two_halo = shell_two_halo * pixel_window_squared[None, :]
    shell_one_halo = shell_one_halo * pixel_window_squared[None, :]
    summed_linear = summed_linear * pixel_window_squared
    summed_two_halo = summed_two_halo * pixel_window_squared
    summed_one_halo = summed_one_halo * pixel_window_squared

    if mean_uncollapsed_counts_per_pixel is None or mean_total_counts_per_pixel is None:
        shell_shot_level = jnp.zeros(z_lo_values.size, dtype=shell_linear.dtype)
        summed_shot_level = jnp.asarray(0.0, dtype=shell_linear.dtype)
    else:
        mean_uncollapsed = jnp.asarray(mean_uncollapsed_counts_per_pixel)
        mean_total = jnp.asarray(mean_total_counts_per_pixel)
        if mean_uncollapsed.shape != (z_lo_values.size,) or mean_total.shape != (z_lo_values.size,):
            raise ValueError("mean count arrays must have shape (n_shell,)")
        shell_shot_level = particle_count_shot_noise(
            mean_uncollapsed,
            mean_total,
            pixel_area_sr,
        )
        summed_shot_level = particle_count_shot_noise(
            jnp.sum(mean_uncollapsed),
            jnp.sum(mean_total),
            pixel_area_sr,
        )

    shell_shot = jnp.broadcast_to(shell_shot_level[:, None], shell_linear.shape)
    summed_shot = jnp.broadcast_to(summed_shot_level, summed_linear.shape)
    shell_clustering = shell_two_halo + shell_one_halo
    summed_clustering = summed_two_halo + summed_one_halo
    return AngularPowerSpectra(
        ell=ell_values,
        shell_linear=shell_linear,
        shell_two_halo=shell_two_halo,
        shell_one_halo=shell_one_halo,
        shell_particle_shot_noise=shell_shot,
        shell_clustering=shell_clustering,
        shell_total=shell_clustering + shell_shot,
        summed_linear=summed_linear,
        summed_two_halo=summed_two_halo,
        summed_one_halo=summed_one_halo,
        summed_particle_shot_noise=summed_shot,
        summed_clustering=summed_clustering,
        summed_total=summed_clustering + summed_shot,
        shell_weights=weights,
        shell_ell_high_ell_start=jnp.asarray(shell_ell_high_ell_start),
        summed_ell_limber_start=jnp.asarray(summed_ell_limber_start),
        ell_limber_start=jnp.asarray(summed_ell_limber_start),
        high_ell_match_shell_relative_error=jnp.asarray(shell_match_error),
        limber_match_summed_relative_error=jnp.asarray(summed_match_error),
        shell_linear_high_ell_mode=jnp.asarray(shell_high_ell_mode),
    )
