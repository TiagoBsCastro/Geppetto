"""JAX occupation and effective satellite kernels.

The occupation equations and named default shape parameters are adapted from
hodpy (Smith et al. 2017), BSD-3-Clause; see docs/licenses/hodpy.txt. M_PIN is
an independent native group mass, never a measured spherical-overdensity mass.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from geppetto.types import Array


class HODShapeParams(NamedTuple):
    """hodpy luminosity-dependent shapes; mass scales are in Msun/h.

    These MXXL-derived numbers are initial functional-shape priors, not a
    PINOCCHIO mass conversion or clustering calibration. All three mass
    scales subsequently receive a common fitted log10(M_PIN) shift.
    """

    mmin_ls: float = 3.91841775e9
    mmin_mt: float = 3.06648452e11
    mmin_am: float = 0.257628181
    m1_ls: float = 3.70558341e9
    m1_mt: float = 4.77678420e12
    m1_am: float = 0.305963831
    m0_a: float = 1.78353177
    m0_b: float = -5.98024172
    alpha_a: float = 0.0982841
    alpha_b: float = 80.27598
    alpha_c: float = 10.0
    sigma_faint: float = 0.02583643
    sigma_bright: float = 0.68126852
    sigma_step: float = 21.05
    sigma_width: float = 2.5


class ThresholdParams(NamedTuple):
    """Broadcastable threshold HOD parameters, with mass scales in log10(Msun/h)."""

    log_mmin: Array
    log_m0: Array
    log_m1: Array
    sigma: Array
    alpha: Array


class SatelliteParams(NamedTuple):
    """Effective radius/velocity assumptions, not measured halo properties.

    R_sat,physical = radius_factor [3 M_PIN/(4 pi Delta rho_mean(z))]^(1/3).
    Concentration is amplitude (M_PIN/pivot)^mass_slope (1+z)^redshift_slope.
    sigma_v,1D = velocity_factor sqrt(G M_PIN/(2 R_sat,physical)).
    """

    overdensity_mean: float = 200.0
    radius_factor: float = 1.0
    concentration_amplitude: float = 5.0
    concentration_mass_pivot: float = 1.e14
    concentration_mass_slope: float = -0.1
    concentration_redshift_slope: float = -0.5
    velocity_factor: float = 1.0
    gravitational_constant: float = 4.300917270e-9  # Mpc (km/s)^2 / Msun


DEFAULT_SATELLITE_PARAMS = SatelliteParams()


def central_probability(log_mass: Array, threshold: ThresholdParams, log_shift: Array = 0.) -> Array:
    """P(central brighter than threshold), differentiable in mass/shape/shift.

    log_mass is log10(M_PIN/[Msun/h]). Uses hodpy's compact spline CDF,
    with support log(M_min) +/- sqrt(6) sigma, rather than Gaussian tails.
    """

    x = (log_mass - threshold.log_mmin - log_shift) / (jnp.sqrt(6.) * threshold.sigma)
    clipped = jnp.clip(x, -1., 1.)
    middle = .5+(clipped-2*clipped**3+1.5*clipped*jnp.abs(clipped)**3)/.75
    # Direct tail polynomials avoid cancellation of tiny bright probabilities.
    value = jnp.where(clipped < -.5, (2/3)*(1+clipped)**4,
                      jnp.where(clipped > .5, 1-(2/3)*(1-clipped)**4, middle))
    return jnp.where(x <= -1, 0., jnp.where(x >= 1, 1., value))


def occupation(log_mass: Array, threshold: ThresholdParams, log_shift: Array = 0.) -> tuple[Array, Array]:
    """Mean cumulative central and satellite counts per native-mass halo.

    Inputs broadcast; outputs have their broadcast shape and are dimensionless.
    The luminosity threshold and fixed calibration table are external inputs.
    Differentiable in the explicit threshold parameters and log mass shift.
    """

    central = central_probability(log_mass, threshold, log_shift)
    excess = (10.**(log_mass-log_shift) - 10.**threshold.log_m0) / 10.**threshold.log_m1
    positive = excess > 0.
    # Avoid evaluating a fractional power at zero in the inactive branch.
    satellite = central * jnp.where(positive, jnp.where(positive, excess, 1.)**threshold.alpha, 0.)
    return central, satellite


def nfw_radial_cdf(radius_fraction: Array, concentration: Array) -> Array:
    """Enclosed probability inside r/R_sat for a truncated 3D NFW number profile.

    Both arguments are dimensionless. Concentration must be positive. The
    profile is isotropic and uses the same cumulative law as hodpy placement.
    """

    x = jnp.clip(radius_fraction, 0., 1.) * concentration
    norm = jnp.log1p(concentration) - concentration / (1+concentration)
    return (jnp.log1p(x) - x/(1+x)) / norm


@jax.custom_jvp
def nfw_inverse_cdf(probability: Array, concentration: Array) -> Array:
    """Inverse radial CDF, r/R_sat, with an implicit concentration derivative.

    Requires 0 <= probability <= 1 and positive concentration. Random-number
    generation is outside this kernel. Endpoint probability derivatives are
    defined as zero; interior derivatives follow implicit differentiation.
    """

    shape = jnp.broadcast_shapes(jnp.shape(probability), jnp.shape(concentration))
    low, high = jnp.zeros(shape), jnp.ones(shape)

    def step(_, bounds):
        lo, hi = bounds
        mid = .5*(lo+hi)
        left = nfw_radial_cdf(mid, concentration) < probability
        return jnp.where(left, mid, lo), jnp.where(left, hi, mid)

    low, high = lax.fori_loop(0, 52, step, (low, high))
    return jnp.where(probability <= 0, 0., jnp.where(probability >= 1, 1., .5*(low+high)))


@nfw_inverse_cdf.defjvp
def _nfw_inverse_jvp(primals, tangents):
    probability, concentration = primals
    dp, dc = tangents
    radius = nfw_inverse_cdf(probability, concentration)
    norm = jnp.log1p(concentration) - concentration/(1+concentration)
    dx = concentration**2 * radius / ((1+concentration*radius)**2 * norm)
    derivative_c = (concentration*radius**2/(1+concentration*radius)**2
                    - probability*concentration/(1+concentration)**2) / norm
    interior = (probability > 0) & (probability < 1)
    derivative = (dp - derivative_c*dc) / jnp.where(interior, dx, 1.)
    return radius, jnp.where(interior, derivative, 0.)


def satellite_scales(
    mass: Array, redshift: Array, mean_density_comoving: Array,
    params: SatelliteParams = DEFAULT_SATELLITE_PARAMS,
) -> tuple[Array, Array, Array]:
    """Return effective (comoving R_sat [Mpc/h], c_sat, sigma_v [km/s]).

    mass is M_PIN in Msun/h, mean density is in (Msun/h)/(Mpc/h)^3.
    Physical radius is comoving radius/(1+z). Nothing is interpreted as an
    observed M200m or R200m. All model parameters are differentiable.
    """

    radius = params.radius_factor * (3*mass/(4*jnp.pi*params.overdensity_mean*mean_density_comoving))**(1/3)
    concentration = params.concentration_amplitude * (mass/params.concentration_mass_pivot)**params.concentration_mass_slope * (1+redshift)**params.concentration_redshift_slope
    sigma = params.velocity_factor * jnp.sqrt(params.gravitational_constant*mass*(1+redshift)/(2*radius))
    return radius, concentration, sigma
