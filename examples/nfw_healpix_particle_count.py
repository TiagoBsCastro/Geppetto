"""Paint an NFW one-halo particle-count map on a small HEALPix grid."""

import jax
import jax.numpy as jnp

from geppetto import ConcentrationParams, LightconeHaloCatalog, paint_lightcone_particle_count_map
from geppetto.io import healpix_pixel_area_sr, healpix_pixel_unit_vectors


def main() -> None:
    nside = 1
    pixel_unit_vectors = jnp.asarray(healpix_pixel_unit_vectors(nside))

    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
        chi=jnp.array([1000.0]),  # Mpc/h
        mass=jnp.array([1.0e14]),  # Msun/h
        redshift=jnp.array([0.3]),
    )

    paint = jax.jit(
        lambda amp: paint_lightcone_particle_count_map(
            pixel_unit_vectors,
            catalog,
            particle_mass_msun_h=1.0e10,
            pixel_area_sr=healpix_pixel_area_sr(nside),
            concentration_params=ConcentrationParams(amplitude=amp),
        )
    )
    counts = paint(5.71)
    print(counts)


if __name__ == "__main__":
    main()
