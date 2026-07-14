"""Small differentiable cosmology helper functions.

The first GEPPETTO release only needs background densities to normalize halo
profiles. Distances can be provided by the caller, which keeps the core painter
independent of any particular background-distance implementation.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax.numpy as jnp

from geppetto.types import Array

# In the internal GEPPETTO unit system, masses are Msun/h and lengths are Mpc/h.
# The numerical value below is therefore also (Msun/h)/(Mpc/h)^3.
RHO_CRIT0_MSUNH_PER_MPCH3 = 2.77536627e11


class Cosmology(NamedTuple):
    """Flat LCDM background parameters used for profile normalization."""

    omega_m: float = 0.315
    h: float = 0.674

    @property
    def omega_l(self) -> float:
        return 1.0 - self.omega_m


def e2_lcdm(z: Array, cosmology: Cosmology) -> Array:
    """Dimensionless Hubble rate squared, ``E(z)^2`` for flat LCDM."""

    return cosmology.omega_m * (1.0 + z) ** 3 + cosmology.omega_l


def omega_m_at_redshift(z: Array, cosmology: Cosmology) -> Array:
    """Return the matter-density fraction ``Omega_m(z)`` for flat LCDM."""

    return cosmology.omega_m * (1.0 + z) ** 3 / e2_lcdm(z, cosmology)


def bryan_norman_virial_overdensity(
    z: Array,
    cosmology: Cosmology,
    reference_density: Literal["critical", "mean"] = "critical",
) -> Array:
    """Return the Bryan--Norman virial overdensity at redshift ``z``.

    The flat-LCDM fit is ``Delta_vir,c = 18*pi^2 + 82*x - 39*x^2`` with
    ``x = Omega_m(z) - 1``. For a mean-matter reference, the returned value is
    ``Delta_vir,m = Delta_vir,c / Omega_m(z)`` so both conventions describe the
    same physical density threshold. The function is differentiable with
    respect to redshift and cosmological parameters.
    """

    omega_m_z = omega_m_at_redshift(z, cosmology)
    x = omega_m_z - 1.0
    delta_critical = 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2
    if reference_density == "critical":
        return delta_critical
    if reference_density == "mean":
        return delta_critical / omega_m_z
    raise ValueError(f"Unknown reference_density={reference_density!r}")


def rho_crit_physical(z: Array, cosmology: Cosmology) -> Array:
    """Critical density at redshift ``z`` in ``(Msun/h)/(Mpc/h)^3`` physical units."""

    return RHO_CRIT0_MSUNH_PER_MPCH3 * e2_lcdm(z, cosmology)


def rho_mean_comoving(cosmology: Cosmology) -> float:
    """Mean matter density in comoving ``(Msun/h)/(Mpc/h)^3`` units."""

    return RHO_CRIT0_MSUNH_PER_MPCH3 * cosmology.omega_m


def halo_radius_delta_comoving(
    mass: Array,
    redshift: Array,
    cosmology: Cosmology,
    overdensity: float = 200.0,
    reference_density: Literal["critical", "mean"] = "critical",
) -> Array:
    """Return the comoving spherical-overdensity radius in Mpc/h.

    ``mass`` is interpreted as the mass enclosed within ``overdensity`` times the
    chosen physical reference density. The returned radius is converted to a
    comoving length so it can be compared directly with PINOCCHIO comoving
    positions.
    """

    if reference_density == "critical":
        rho_ref_phys = rho_crit_physical(redshift, cosmology)
        r_phys = (3.0 * mass / (4.0 * jnp.pi * overdensity * rho_ref_phys)) ** (1.0 / 3.0)
        return r_phys * (1.0 + redshift)
    if reference_density == "mean":
        # rho_m,phys(z) = rho_m,com * (1 + z)^3; converting to comoving radius
        # cancels the redshift factor.
        rho_ref_com = rho_mean_comoving(cosmology)
        return (3.0 * mass / (4.0 * jnp.pi * overdensity * rho_ref_com)) ** (1.0 / 3.0)
    raise ValueError(f"Unknown reference_density={reference_density!r}")
