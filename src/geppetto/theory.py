"""Matter-power and angular-spectrum theory for PINOCCHIO map validation.

The differentiable part of this module implements a linear-plus-one-halo
matter-power model. PINOCCHIO supplies the tabulated linear spectrum and
measured halo mass functions; GEPPETTO supplies the NFW mass definition and
concentration relation used by the map painter.

All masses are ``Msun/h``, comoving distances are ``Mpc/h``, wavenumbers are
``h/Mpc``, three-dimensional power spectra are ``(Mpc/h)^3``, and angular
power spectra are dimensionless. File parsing and HEALPix operations live
outside this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

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


class HaloMassFunctionTable(NamedTuple):
    """Measured PINOCCHIO mass functions on a common mass grid.

    ``scale_factor`` is increasing, ``log_mass_msun_h`` is the natural
    logarithm of mass in ``Msun/h``, and ``dndlnm_mpc_h3`` has shape
    ``(n_scale_factor, n_mass)`` in ``(Mpc/h)^-3``.
    """

    scale_factor: Array
    log_mass_msun_h: Array
    dndlnm_mpc_h3: Array


class QuadratureRule(NamedTuple):
    """Nodes and weights for integration over the interval ``[-1, 1]``."""

    nodes: Array
    weights: Array


class AngularPowerSpectra(NamedTuple):
    """Per-shell and count-weighted-sum angular power spectra.

    Every shell array has shape ``(n_shell, n_ell)``. Summed arrays have shape
    ``(n_ell,)``. ``linear`` and ``one_halo`` include the supplied HEALPix
    pixel window. Particle shot noise is kept white in the pixel-count map
    convention.
    """

    ell: Array
    shell_linear: Array
    shell_one_halo: Array
    shell_particle_shot_noise: Array
    shell_clustering: Array
    shell_total: Array
    summed_linear: Array
    summed_one_halo: Array
    summed_particle_shot_noise: Array
    summed_clustering: Array
    summed_total: Array
    shell_weights: Array
    shell_ell_limber_start: Array
    summed_ell_limber_start: Array
    ell_limber_start: Array
    limber_match_shell_relative_error: Array
    limber_match_summed_relative_error: Array


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
) -> Array:
    """Return linear matter power in ``(Mpc/h)^3``.

    The caller must validate that requested wavenumbers and redshifts lie
    within the PINOCCHIO tables. Values are clipped at table boundaries inside
    this JAX-compatible kernel to avoid dynamic host-side exceptions.
    """

    k = jnp.asarray(k_h_mpc)
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
    return jnp.where(jnp.abs(value) < 1.0e-3, series, direct)


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
    """Return measured-HMF one-halo power in ``(Mpc/h)^3``.

    When ``theta_resolution_rad`` is supplied, halos below the map's angular
    NGP threshold use ``u=1``. The branch depends only on mass, redshift,
    distance, and profile support, never concentration. Supersampled and native
    resolved halos share the continuum NFW transform.
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
    integrand = dndlnm[None, :] * (mass[None, :] / mean_density) ** 2 * profile**2
    result = _trapezoid_last_axis(integrand, mass_function.log_mass_msun_h)
    return result[0] if scalar_k else result


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


def limber_shell_cls(
    ell: Array,
    z_lo: float,
    z_hi: float,
    linear_theory: LinearTheoryTable,
    mass_function: HaloMassFunctionTable,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams = DEFAULT_NFW_PROFILE_PARAMS,
    *,
    theta_resolution_rad: float | None = None,
    radial_quadrature: QuadratureRule | None = None,
    profile_quadrature: QuadratureRule | None = None,
) -> tuple[Array, Array]:
    """Return Limber linear and one-halo ``C_ell`` for one count shell.

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

    def node_power(inputs: tuple[Array, Array]) -> tuple[Array, Array]:
        chi_node, redshift_node = inputs
        k = (ell_values + 0.5) / chi_node
        linear = linear_matter_power(k, redshift_node, linear_theory)
        linear = jnp.where(
            (k >= linear_theory.k_h_mpc[0]) & (k <= linear_theory.k_h_mpc[-1]),
            linear,
            0.0,
        )
        one_halo = one_halo_matter_power(
            k,
            redshift_node,
            linear_theory,
            mass_function,
            concentration_params,
            profile_params,
            theta_resolution_rad=theta_resolution_rad,
            profile_quadrature=profile_quadrature,
        )
        return linear, one_halo

    linear_nodes, one_halo_nodes = lax.map(node_power, (chi, redshift))
    return (
        jnp.sum(radial_prefactor[:, None] * linear_nodes, axis=0),
        jnp.sum(radial_prefactor[:, None] * one_halo_nodes, axis=0),
    )


class _ExactProjectionState(NamedTuple):
    log_k_table: np.ndarray
    log_power_table: np.ndarray
    chi_nodes: np.ndarray
    transfer_weight: np.ndarray
    weights: np.ndarray
    shell_midpoint: np.ndarray
    shell_width: np.ndarray
    k_table_max: float
    log_k_min: float
    relative_tolerance: float
    radial_tail_periods: float


_EXACT_PROJECTION_STATE: _ExactProjectionState | None = None


def _integrate_exact_multipole(
    ell_value: int,
    state: _ExactProjectionState,
) -> tuple[np.ndarray, float]:
    """Integrate one exact multipole using process-local read-only state."""

    from scipy.integrate import quad
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

    def shell_integrand(log_k: float, shell_index: int) -> float:
        k, prefactor = power_at_log_k(log_k)
        bessel = spherical_jn(ell_value, k * state.chi_nodes[shell_index])
        transfer = np.sum(state.transfer_weight[shell_index] * bessel)
        return float(prefactor * transfer**2)

    shell_integrated = np.empty(state.chi_nodes.shape[0], dtype=np.float64)
    for shell_index, shell_limit in enumerate(shell_k_max):
        shell_integrated[shell_index], _ = quad(
            shell_integrand,
            state.log_k_min,
            np.log(shell_limit),
            args=(shell_index,),
            epsabs=1.0e-14,
            epsrel=state.relative_tolerance,
            limit=2000,
        )

    def summed_integrand(log_k: float) -> float:
        k, prefactor = power_at_log_k(log_k)
        bessel = spherical_jn(ell_value, k * state.chi_nodes)
        transfer = np.sum(state.transfer_weight * bessel, axis=1)
        transfer = np.where(k <= shell_k_max, transfer, 0.0)
        summed_transfer = np.sum(state.weights * transfer)
        return float(prefactor * summed_transfer**2)

    summed_integrated, _ = quad(
        summed_integrand,
        state.log_k_min,
        np.log(k_max),
        epsabs=1.0e-14,
        epsrel=state.relative_tolerance,
        limit=2000,
    )
    return shell_integrated, float(summed_integrated)


def _initialize_exact_projection_worker(state: _ExactProjectionState) -> None:
    global _EXACT_PROJECTION_STATE
    _EXACT_PROJECTION_STATE = state


def _integrate_exact_multipole_worker(ell_value: int) -> tuple[np.ndarray, float]:
    if _EXACT_PROJECTION_STATE is None:
        raise RuntimeError("exact projection worker was not initialized")
    return _integrate_exact_multipole(ell_value, _EXACT_PROJECTION_STATE)


def exact_linear_shell_cls(
    ell: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    linear_theory: LinearTheoryTable,
    *,
    shell_weights: np.ndarray | None = None,
    radial_order: int = 512,
    radial_tail_periods: float = 256.0,
    relative_tolerance: float = 1.0e-4,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact low-ell linear shell autos and weighted-sum spectrum.

    This concentration-independent orchestration path uses SciPy's spherical
    Bessel functions and adaptive scalar quadrature. Shell autos use
    shell-specific wavenumber cutoffs, while the weighted sum retains
    cross-shell correlations up to each shell's cutoff. The radial tail spans
    at least 40 oscillation periods and grows with transverse wavenumber up to
    ``radial_tail_periods``. Independent multipoles are evaluated in
    ``workers`` spawned processes. Install
    ``geppetto[theory]`` to use it. The differentiable one-halo and Limber
    kernels do not depend on SciPy.
    """

    try:
        __import__("scipy.integrate")
        __import__("scipy.special")
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "exact_linear_shell_cls requires scipy; install geppetto[theory]"
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
    k_table = np.asarray(linear_theory.k_h_mpc, dtype=np.float64)
    power_table = np.asarray(linear_theory.power_mpc_h3, dtype=np.float64)
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

    shell_result = np.empty((n_shell, ell_values.size), dtype=np.float64)
    summed_result = np.empty(ell_values.size, dtype=np.float64)
    shell_midpoint = 0.5 * (chi_lo + chi_hi)
    shell_width = chi_hi - chi_lo
    state = _ExactProjectionState(
        log_k_table=log_k_table,
        log_power_table=log_power_table,
        chi_nodes=chi_nodes,
        transfer_weight=transfer_weight,
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
        for ell_index, (shell_integrated, sum_integrated) in enumerate(
            integrated_multipoles
        ):
            shell_result[:, ell_index] = shell_integrated
            summed_result[ell_index] = sum_integrated
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
                for ell_index, (shell_integrated, sum_integrated) in enumerate(
                    integrated_multipoles
                ):
                    shell_result[:, ell_index] = shell_integrated
                    summed_result[ell_index] = sum_integrated
        finally:
            for name, previous_value in previous_affinity.items():
                if previous_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous_value

    return shell_result, summed_result


def particle_count_shot_noise(
    mean_uncollapsed_counts_per_pixel: Array,
    mean_total_counts_per_pixel: Array,
    pixel_area_sr: float,
) -> Array:
    """Return white particle shot noise for total count-overdensity maps."""

    uncollapsed = jnp.asarray(mean_uncollapsed_counts_per_pixel)
    total = jnp.asarray(mean_total_counts_per_pixel)
    return pixel_area_sr * uncollapsed / total**2


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
    """Select the first numerically matched exact-to-Limber transition.

    Arrays contain the same contiguous multipoles. A candidate is accepted
    only when every shell and the count-weighted sum agree for the requested
    number of consecutive multipoles. Returned errors are maxima over that
    confirmation interval.
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
    shell_error = np.abs(exact_shell_values - limber_shell_values) / np.maximum(
        shell_scale, tiny
    )
    sum_error = np.abs(exact_sum_values - limber_sum_values) / np.maximum(sum_scale, tiny)
    worst_error = np.maximum(np.max(shell_error, axis=0), sum_error)
    for start in range(0, ell_values.size - consecutive_multipoles + 1):
        stop = start + consecutive_multipoles
        if np.all(worst_error[start:stop] <= relative_tolerance):
            return (
                int(ell_values[start]),
                np.max(shell_error[:, start:stop], axis=1),
                float(np.max(sum_error[start:stop])),
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
    """Select exact-to-Limber transitions independently for each spectrum.

    A shell does not need to enter its valid Limber regime at the same
    multipole as every other shell. The returned shell transition array uses
    ``-1`` for spectra that have not matched. Errors for matched spectra are
    maxima over their accepted confirmation windows; unmatched errors are
    maxima over the final available window, which diagnoses the failure near
    the exact-projection cap rather than the expected low-multipole mismatch.
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
    sum_error_values = np.abs(exact_sum_values - limber_sum_values) / np.maximum(
        sum_scale, tiny
    )
    n_candidate = ell_values.size - consecutive_multipoles + 1
    final_start = max(0, ell_values.size - consecutive_multipoles)

    shell_transition = np.full(exact_shell_values.shape[0], -1, dtype=np.int64)
    shell_error = np.empty(exact_shell_values.shape[0], dtype=np.float64)
    for shell_index, errors in enumerate(shell_error_values):
        accepted_start = next(
            (
                start
                for start in range(max(0, n_candidate))
                if np.all(errors[start : start + consecutive_multipoles] <= relative_tolerance)
            ),
            None,
        )
        if accepted_start is None:
            shell_error[shell_index] = float(np.max(errors[final_start:]))
        else:
            shell_transition[shell_index] = int(ell_values[accepted_start])
            shell_error[shell_index] = float(
                np.max(errors[accepted_start : accepted_start + consecutive_multipoles])
            )

    summed_start = next(
        (
            start
            for start in range(max(0, n_candidate))
            if np.all(
                sum_error_values[start : start + consecutive_multipoles] <= relative_tolerance
            )
        ),
        None,
    )
    if summed_start is None:
        summed_transition = None
        summed_error = float(np.max(sum_error_values[final_start:]))
    else:
        summed_transition = int(ell_values[summed_start])
        summed_error = float(
            np.max(sum_error_values[summed_start : summed_start + consecutive_multipoles])
        )
    return shell_transition, summed_transition, shell_error, summed_error


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
    exact_batch_evaluator: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    radial_order: int = 64,
    exact_radial_order: int = 512,
    exact_radial_tail_periods: float = 256.0,
    profile_order: int = 64,
    exact_relative_tolerance: float = 1.0e-4,
) -> AngularPowerSpectra:
    """Compute hybrid exact/Limber spectra for disjoint PINOCCHIO shells.

    Exact spherical-Bessel projection is used until each shell and the
    weighted-sum spectrum independently agree with Limber within
    ``limber_match_rtol`` for ``limber_match_width`` consecutive multipoles.
    The search is bounded by ``ell_exact_cap``. A zero cap explicitly selects
    Limber at all multipoles. ``exact_batch_evaluator`` may provide externally
    cached exact batches; it receives only the requested multipole vector.
    One-halo power always uses Limber. This wrapper performs
    concentration-independent SciPy orchestration; use :func:`limber_shell_cls`
    directly inside concentration-gradient transformations.
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
    ):
        raise ValueError("Limber and exact projection controls must be positive")
    if ell_exact_cap > 0 and np.any(np.diff(ell_numpy) != 1):
        raise ValueError("adaptive exact-to-Limber selection requires contiguous multipoles")

    weights = jnp.asarray(shell_weights)
    if weights.shape != (z_lo_values.size,):
        raise ValueError("shell_weights must have shape (n_shell,)")
    weights = weights / jnp.sum(weights)
    radial_quadrature = gauss_legendre_rule(radial_order)
    profile_quadrature = gauss_legendre_rule(profile_order)
    limber_results = [
        limber_shell_cls(
            ell_values,
            float(lo),
            float(hi),
            linear_theory,
            mass_function,
            concentration_params,
            shell_profile,
            theta_resolution_rad=theta_resolution_rad,
            radial_quadrature=radial_quadrature,
            profile_quadrature=profile_quadrature,
        )
        for lo, hi, shell_profile in zip(z_lo_values, z_hi_values, profiles, strict=True)
    ]
    shell_linear = jnp.stack([result[0] for result in limber_results])
    shell_one_halo = jnp.stack([result[1] for result in limber_results])
    summed_linear = jnp.sum(weights[:, None] ** 2 * shell_linear, axis=0)
    summed_one_halo = jnp.sum(weights[:, None] ** 2 * shell_one_halo, axis=0)

    shell_ell_limber_start = np.full(z_lo_values.size, int(ell_numpy[0]), dtype=np.int64)
    summed_ell_limber_start = int(ell_numpy[0])
    shell_match_error = np.full(z_lo_values.size, np.nan, dtype=np.float64)
    summed_match_error = np.nan
    if ell_exact_cap > 0 and np.any(ell_numpy <= ell_exact_cap):
        exact_indices_all = np.flatnonzero(ell_numpy <= ell_exact_cap)
        exact_shell_blocks: list[np.ndarray] = []
        exact_sum_blocks: list[np.ndarray] = []
        shell_transition = np.full(z_lo_values.size, -1, dtype=np.int64)
        summed_transition: int | None = None
        for batch_start in range(0, exact_indices_all.size, exact_batch_size):
            batch_indices = exact_indices_all[batch_start : batch_start + exact_batch_size]
            batch_ell_min = int(ell_numpy[batch_indices[0]])
            batch_ell_max = int(ell_numpy[batch_indices[-1]])
            if exact_batch_callback is not None:
                exact_batch_callback("start", batch_ell_min, batch_ell_max)
            if exact_batch_evaluator is None:
                exact_shell_batch, exact_sum_batch = exact_linear_shell_cls(
                    ell_numpy[batch_indices],
                    z_lo_values,
                    z_hi_values,
                    linear_theory,
                    shell_weights=np.asarray(weights),
                    radial_order=exact_radial_order,
                    radial_tail_periods=exact_radial_tail_periods,
                    relative_tolerance=exact_relative_tolerance,
                    workers=exact_workers,
                )
            else:
                exact_shell_batch, exact_sum_batch = exact_batch_evaluator(
                    ell_numpy[batch_indices]
                )
            exact_shell_batch = np.asarray(exact_shell_batch, dtype=np.float64)
            exact_sum_batch = np.asarray(exact_sum_batch, dtype=np.float64)
            expected_shell_shape = (z_lo_values.size, batch_indices.size)
            if exact_shell_batch.shape != expected_shell_shape or exact_sum_batch.shape != (
                batch_indices.size,
            ):
                raise ValueError("exact batch evaluator returned inconsistent spectrum shapes")
            if not np.all(np.isfinite(exact_shell_batch)) or not np.all(
                np.isfinite(exact_sum_batch)
            ):
                raise ValueError("exact batch evaluator returned non-finite spectra")
            if exact_batch_callback is not None:
                exact_batch_callback("complete", batch_ell_min, batch_ell_max)
            exact_shell_blocks.append(exact_shell_batch)
            exact_sum_blocks.append(exact_sum_batch)
            exact_count = sum(block.shape[1] for block in exact_shell_blocks)
            if exact_count < limber_match_width or ell_numpy[-1] <= ell_exact_cap:
                continue
            compared_indices = exact_indices_all[:exact_count]
            (
                shell_transition,
                summed_transition,
                shell_match_error,
                summed_match_error,
            ) = select_independent_limber_transitions(
                ell_numpy[compared_indices],
                np.concatenate(exact_shell_blocks, axis=1),
                np.concatenate(exact_sum_blocks),
                np.asarray(shell_linear)[:, compared_indices],
                np.asarray(summed_linear)[compared_indices],
                relative_tolerance=limber_match_rtol,
                consecutive_multipoles=limber_match_width,
            )
            if np.all(shell_transition >= 0) and summed_transition is not None:
                break

        exact_shell = np.concatenate(exact_shell_blocks, axis=1)
        exact_sum = np.concatenate(exact_sum_blocks)
        exact_indices = exact_indices_all[: exact_sum.size]
        if ell_numpy[-1] <= ell_exact_cap:
            shell_ell_limber_start.fill(int(ell_numpy[-1]) + 1)
            summed_ell_limber_start = int(ell_numpy[-1]) + 1
        else:
            if np.any(shell_transition < 0) or summed_transition is None:
                (
                    shell_transition,
                    summed_transition,
                    shell_match_error,
                    summed_match_error,
                ) = select_independent_limber_transitions(
                    ell_numpy[exact_indices],
                    exact_shell,
                    exact_sum,
                    np.asarray(shell_linear)[:, exact_indices],
                    np.asarray(summed_linear)[exact_indices],
                    relative_tolerance=limber_match_rtol,
                    consecutive_multipoles=limber_match_width,
                )
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
                    "an exact and Limber linear projection did not converge before "
                    f"ell_exact_cap={ell_exact_cap}: {worst_label}, "
                    f"final_window={final_window_start}-{ell_numpy[exact_indices[-1]]}, "
                    f"maximum_relative_error={worst_error:.6g}"
                )
            shell_ell_limber_start = shell_transition
            summed_ell_limber_start = int(summed_transition)
        for shell_index, transition in enumerate(shell_ell_limber_start):
            use_exact = exact_indices[ell_numpy[exact_indices] < transition]
            if use_exact.size:
                exact_lookup = np.searchsorted(exact_indices, use_exact)
                shell_linear = shell_linear.at[shell_index, jnp.asarray(use_exact)].set(
                    jnp.asarray(
                        exact_shell[shell_index, exact_lookup], dtype=shell_linear.dtype
                    )
                )
        use_exact_sum = exact_indices[
            ell_numpy[exact_indices] < summed_ell_limber_start
        ]
        if use_exact_sum.size:
            exact_lookup = np.searchsorted(exact_indices, use_exact_sum)
            summed_linear = summed_linear.at[jnp.asarray(use_exact_sum)].set(
                jnp.asarray(exact_sum[exact_lookup], dtype=summed_linear.dtype)
            )

    if pixel_window is None:
        pixel_window_values = jnp.ones_like(ell_values, dtype=shell_linear.dtype)
    else:
        pixel_window_values = jnp.asarray(pixel_window)
        if pixel_window_values.shape != ell_values.shape:
            raise ValueError("pixel_window must have shape (n_ell,)")
    pixel_window_squared = pixel_window_values**2
    shell_linear = shell_linear * pixel_window_squared[None, :]
    shell_one_halo = shell_one_halo * pixel_window_squared[None, :]
    summed_linear = summed_linear * pixel_window_squared
    summed_one_halo = summed_one_halo * pixel_window_squared

    if mean_uncollapsed_counts_per_pixel is None or mean_total_counts_per_pixel is None:
        shell_shot_level = jnp.zeros(z_lo_values.size, dtype=shell_linear.dtype)
        summed_shot_level = jnp.asarray(0.0, dtype=shell_linear.dtype)
    else:
        mean_uncollapsed = jnp.asarray(mean_uncollapsed_counts_per_pixel)
        mean_total = jnp.asarray(mean_total_counts_per_pixel)
        if mean_uncollapsed.shape != (z_lo_values.size,) or mean_total.shape != (
            z_lo_values.size,
        ):
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
    shell_clustering = shell_linear + shell_one_halo
    summed_clustering = summed_linear + summed_one_halo
    return AngularPowerSpectra(
        ell=ell_values,
        shell_linear=shell_linear,
        shell_one_halo=shell_one_halo,
        shell_particle_shot_noise=shell_shot,
        shell_clustering=shell_clustering,
        shell_total=shell_clustering + shell_shot,
        summed_linear=summed_linear,
        summed_one_halo=summed_one_halo,
        summed_particle_shot_noise=summed_shot,
        summed_clustering=summed_clustering,
        summed_total=summed_clustering + summed_shot,
        shell_weights=weights,
        shell_ell_limber_start=jnp.asarray(shell_ell_limber_start),
        summed_ell_limber_start=jnp.asarray(summed_ell_limber_start),
        ell_limber_start=jnp.asarray(summed_ell_limber_start),
        limber_match_shell_relative_error=jnp.asarray(shell_match_error),
        limber_match_summed_relative_error=jnp.asarray(summed_match_error),
    )
