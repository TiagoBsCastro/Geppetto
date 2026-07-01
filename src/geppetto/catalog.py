"""Halo-catalogue containers.

The containers are intentionally small ``NamedTuple`` objects. JAX treats them as
pytrees, so functions using them can be differentiated and transformed with
``jax.jit``, ``jax.grad``, ``jax.vmap`` and ``jax.lax.scan``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from geppetto.types import Array


class HaloCatalog(NamedTuple):
    """Minimal halo catalogue for comoving-box painting.

    Parameters
    ----------
    position:
        Halo Cartesian positions with shape ``(n_halo, 3)`` in comoving Mpc/h.
    mass:
        Halo mass with shape ``(n_halo,)`` in Msun/h. The default profile
        interpretation is ``M_200c`` unless the user changes the profile
        prescription.
    redshift:
        Halo redshift with shape ``(n_halo,)``. For snapshot boxes this can be a
        constant array.
    """

    position: Array
    mass: Array
    redshift: Array

    @property
    def size(self) -> int:
        return int(self.mass.shape[0])


class LightconeHaloCatalog(NamedTuple):
    """Minimal halo catalogue for PLC/lightcone painting.

    Parameters
    ----------
    unit_vector:
        Unit vector pointing to each halo, shape ``(n_halo, 3)``.
    chi:
        Comoving radial distance in Mpc/h, shape ``(n_halo,)``.
    mass:
        Halo mass in Msun/h, shape ``(n_halo,)``.
    redshift:
        True halo redshift, shape ``(n_halo,)``.
    """

    unit_vector: Array
    chi: Array
    mass: Array
    redshift: Array

    @property
    def position(self) -> Array:
        """Cartesian comoving position, shape ``(n_halo, 3)``."""

        return self.chi[:, None] * self.unit_vector

    @property
    def size(self) -> int:
        return int(self.mass.shape[0])


def unit_vectors_from_angles(theta: Array, phi: Array) -> Array:
    """Convert spherical angles to unit vectors.

    Parameters
    ----------
    theta:
        Colatitude in radians, HEALPix convention.
    phi:
        Longitude in radians.
    """

    sin_theta = jnp.sin(theta)
    return jnp.stack(
        [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), jnp.cos(theta)], axis=-1
    )


def from_spherical_lightcone(theta: Array, phi: Array, chi: Array, mass: Array, redshift: Array) -> LightconeHaloCatalog:
    """Build a lightcone catalogue from HEALPix-style angles and distances."""

    return LightconeHaloCatalog(
        unit_vector=unit_vectors_from_angles(theta, phi),
        chi=chi,
        mass=mass,
        redshift=redshift,
    )
