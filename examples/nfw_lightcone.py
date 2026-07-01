"""Paint projected NFW surface density on supplied lightcone pixel vectors."""

import jax
import jax.numpy as jnp

from geppetto import ConcentrationParams, LightconeHaloCatalog, paint_lightcone_surface_density


def main() -> None:
    pixel_unit_vectors = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.045, 0.0],
            [0.995, 0.100, 0.0],
        ]
    )
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]

    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
        chi=jnp.array([1000.0]),
        mass=jnp.array([1.0e14]),
        redshift=jnp.array([0.3]),
    )

    paint = jax.jit(
        lambda amp: paint_lightcone_surface_density(
            pixel_unit_vectors,
            catalog,
            concentration_params=ConcentrationParams(amplitude=amp),
        )
    )
    sigma = paint(1.0)
    print(sigma)


if __name__ == "__main__":
    main()
