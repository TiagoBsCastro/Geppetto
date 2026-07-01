"""Paint a small NFW one-halo density grid in a periodic comoving box."""

import jax
import jax.numpy as jnp

from geppetto import Cosmology, HaloCatalog, duffy08_all_200c, paint_box_density_grid


def main() -> None:
    catalog = HaloCatalog(
        position=jnp.array([[50.0, 50.0, 50.0], [25.0, 25.0, 30.0]]),
        mass=jnp.array([1.0e14, 5.0e13]),
        redshift=jnp.array([0.0, 0.0]),
    )

    paint = jax.jit(
        lambda amplitude: paint_box_density_grid(
            catalog,
            box_size=100.0,
            nmesh=32,
            cosmology=Cosmology(),
            concentration_params=duffy08_all_200c()._replace(amplitude=amplitude),
            periodic=True,
            chunk_size=1,
        )
    )

    grid = paint(5.71)
    print(grid.shape, grid.min(), grid.max())


if __name__ == "__main__":
    main()
