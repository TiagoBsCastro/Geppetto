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

from collections.abc import Sequence
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


def exact_linear_shell_cls(
    ell: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    linear_theory: LinearTheoryTable,
    *,
    shell_weights: np.ndarray | None = None,
    radial_order: int = 64,
    relative_tolerance: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact low-ell linear shell autos and weighted-sum spectrum.

    This concentration-independent orchestration path uses SciPy's spherical
    Bessel functions and adaptive vector quadrature. Install ``geppetto[theory]``
    to use it. The differentiable one-halo and Limber kernels do not depend on
    SciPy.
    """

    try:
        from scipy.integrate import quad_vec
        from scipy.special import spherical_jn
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
    log_k_min = float(log_k_table[0])
    shell_midpoint = 0.5 * (chi_lo + chi_hi)
    shell_width = chi_hi - chi_lo

    for ell_index, ell_value in enumerate(ell_values):
        if ell_value < 0:
            raise ValueError("ell values must be non-negative")
        # A finite top-hat shell suppresses radial modes far above 1/DeltaChi.
        # Truncating after 40 radial oscillations avoids aliasing those
        # cancellations on the fixed radial quadrature while retaining a
        # converged low-ell transfer integral.
        transverse_k = (ell_value + 0.5) / shell_midpoint
        radial_tail_k = 40.0 * np.pi / shell_width
        k_max = min(float(k_table[-1]), float(np.max(transverse_k + radial_tail_k)))
        log_k_max = np.log(k_max)

        def integrand(log_k: float, ell_order: int = int(ell_value)) -> np.ndarray:
            k = np.exp(log_k)
            power = np.exp(np.interp(log_k, log_k_table, log_power_table))
            bessel = spherical_jn(ell_order, k * chi_nodes)
            transfer = np.sum(transfer_weight * bessel, axis=1)
            summed_transfer = np.sum(weights * transfer)
            prefactor = (2.0 / np.pi) * k**3 * power
            return prefactor * np.concatenate([transfer**2, [summed_transfer**2]])

        integrated, _ = quad_vec(
            integrand,
            log_k_min,
            log_k_max,
            epsabs=1.0e-14,
            epsrel=relative_tolerance,
            limit=2000,
        )
        shell_result[:, ell_index] = integrated[:-1]
        summed_result[ell_index] = integrated[-1]

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
    ell_exact_max: int = 100,
    radial_order: int = 64,
    profile_order: int = 64,
    exact_relative_tolerance: float = 1.0e-4,
) -> AngularPowerSpectra:
    """Compute hybrid exact/Limber spectra for disjoint PINOCCHIO shells.

    Exact spherical-Bessel projection replaces the linear Limber result through
    ``ell_exact_max``. One-halo power always uses Limber. This wrapper performs
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

    exact_mask = np.asarray(ell_values) <= ell_exact_max
    if np.any(exact_mask):
        exact_shell, exact_sum = exact_linear_shell_cls(
            np.asarray(ell_values)[exact_mask],
            z_lo_values,
            z_hi_values,
            linear_theory,
            shell_weights=np.asarray(weights),
            radial_order=radial_order,
            relative_tolerance=exact_relative_tolerance,
        )
        exact_indices = jnp.asarray(np.flatnonzero(exact_mask))
        shell_linear = shell_linear.at[:, exact_indices].set(jnp.asarray(exact_shell))
        summed_linear = summed_linear.at[exact_indices].set(jnp.asarray(exact_sum))

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
    )
