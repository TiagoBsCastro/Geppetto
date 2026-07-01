"""Differentiable geometry helpers for boxes and lightcones."""

from __future__ import annotations

import jax.numpy as jnp

from geppetto.types import Array


def periodic_displacement(dx: Array, box_size: float | None) -> Array:
    """Return minimum-image displacement for a periodic cubic box."""

    if box_size is None:
        return dx
    return dx - box_size * jnp.round(dx / box_size)


def pairwise_radius(points: Array, centres: Array, box_size: float | None = None) -> Array:
    """Pairwise radius between target points and centres.

    Returns an array with shape ``(n_point, n_centre)``.
    """

    dx = points[:, None, :] - centres[None, :, :]
    dx = periodic_displacement(dx, box_size)
    return jnp.linalg.norm(dx, axis=-1)


def box_grid_positions(box_size: float, nmesh: int) -> Array:
    """Return cubic-cell centre positions, shape ``(nmesh**3, 3)``."""

    edges = (jnp.arange(nmesh) + 0.5) * (box_size / nmesh)
    xx, yy, zz = jnp.meshgrid(edges, edges, edges, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def angular_separation_from_unit_vectors(a: Array, b: Array) -> Array:
    """Angular separation between two sets of unit vectors.

    Returns an array with shape ``(a.shape[0], b.shape[0])``.
    """

    cosang = jnp.clip(a @ b.T, -1.0, 1.0)
    return jnp.arccos(cosang)


def transverse_distance_from_unit_vectors(pixel_unit_vectors: Array, halo_unit_vectors: Array, halo_chi: Array) -> Array:
    """Approximate transverse comoving separation for lightcone projection.

    The chord form ``chi * sqrt(2 * (1 - cos(theta)))`` is stable for small
    angular separations and avoids computing ``arccos``. For the one-halo term
    this is usually sufficient because the profile support is compact.
    """

    cosang = jnp.clip(pixel_unit_vectors @ halo_unit_vectors.T, -1.0, 1.0)
    chord = jnp.sqrt(jnp.maximum(2.0 * (1.0 - cosang), 0.0))
    return chord * halo_chi[None, :]
