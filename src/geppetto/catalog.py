"""Halo-catalogue containers.

The containers are intentionally small pytrees, so functions using them can be
differentiated and transformed with ``jax.jit``, ``jax.grad``, ``jax.vmap`` and
``jax.lax.scan``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LightconeSparseStencil:
    """Sparse lightcone halo-pixel stencil for one-halo painting.

    Parameters
    ----------
    pix_id:
        Output pixel index for each retained halo-pixel pair, shape ``(n_pair,)``.
    halo_id:
        Halo index into a ``LightconeHaloCatalog`` for each retained pair, shape
        ``(n_pair,)``.
    r_perp:
        Transverse comoving separation for each pair in ``Mpc/h``, shape
        ``(n_pair,)``. The stencil builder fixes these geometry values outside
        differentiable painter kernels.
    n_pix:
        Number of pixels in the output one-dimensional map.
    pair_weight:
        Optional dimensionless contribution weight for each retained pair,
        shape ``(n_pair,)``. ``None`` is equivalent to unit weights. Zero
        weights support shape padding without changing painted maps.
    """

    pix_id: Array
    halo_id: Array
    r_perp: Array
    n_pix: int
    pair_weight: Array | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_pix", int(self.n_pix))

    def tree_flatten(self) -> tuple[tuple[Array, Array, Array, Array | None], int]:
        """Keep ``n_pix`` static for ``jax.jit`` output-shape construction."""

        return (self.pix_id, self.halo_id, self.r_perp, self.pair_weight), self.n_pix

    @classmethod
    def tree_unflatten(
        cls, n_pix: int, children: tuple[Array, Array, Array, Array | None]
    ) -> LightconeSparseStencil:
        pix_id, halo_id, r_perp, pair_weight = children
        return cls(
            pix_id=pix_id,
            halo_id=halo_id,
            r_perp=r_perp,
            n_pix=n_pix,
            pair_weight=pair_weight,
        )

    @property
    def size(self) -> int:
        return int(self.r_perp.shape[0])


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
