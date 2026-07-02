import jax
import jax.numpy as jnp

from geppetto import (
    ConcentrationParams,
    Cosmology,
    HaloCatalog,
    LightconeHaloCatalog,
    NFWProfileParams,
    density_at_points,
    density_at_points_chunked,
    paint_box_density_grid,
    paint_lightcone_particle_count_map,
    paint_lightcone_surface_density,
    paint_lightcone_surface_density_sparse,
)
from geppetto.io import build_lightcone_sparse_stencil


def test_density_at_points_shape_and_grad():
    points = jnp.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]])
    catalog = HaloCatalog(
        position=jnp.array([[50.0, 50.0, 50.0]]),
        mass=jnp.array([1.0e14]),
        redshift=jnp.array([0.0]),
    )

    def objective(amplitude):
        return jnp.sum(
            density_at_points(
                points,
                catalog,
                Cosmology(),
                ConcentrationParams(amplitude=amplitude),
                periodic_box_size=100.0,
            )
        )

    rho = density_at_points(points, catalog, periodic_box_size=100.0)
    assert rho.shape == (2,)
    assert jnp.isfinite(jax.grad(objective)(5.71))


def test_chunked_matches_unchunked():
    points = jnp.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]])
    catalog = HaloCatalog(
        position=jnp.array([[50.0, 50.0, 50.0], [60.0, 60.0, 60.0], [10.0, 10.0, 10.0]]),
        mass=jnp.array([1.0e14, 5.0e13, 2.0e13]),
        redshift=jnp.array([0.0, 0.0, 0.0]),
    )
    direct = density_at_points(points, catalog, periodic_box_size=100.0)
    chunked = density_at_points_chunked(points, catalog, periodic_box_size=100.0, chunk_size=2)
    assert jnp.allclose(direct, chunked, rtol=1.0e-5, atol=1.0e-5)


def test_paint_box_density_grid_shape():
    catalog = HaloCatalog(
        position=jnp.array([[50.0, 50.0, 50.0]]),
        mass=jnp.array([1.0e14]),
        redshift=jnp.array([0.0]),
    )
    grid = paint_box_density_grid(catalog, box_size=100.0, nmesh=8, periodic=True, chunk_size=1)
    assert grid.shape == (8, 8, 8)
    assert jnp.all(jnp.isfinite(grid))


def test_lightcone_surface_density_shape_and_grad():
    pixel_unit_vectors = jnp.array([[1.0, 0.0, 0.0], [0.999, 0.045, 0.0]])
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
        chi=jnp.array([1000.0]),
        mass=jnp.array([1.0e14]),
        redshift=jnp.array([0.3]),
    )

    def objective(amplitude):
        return jnp.sum(
            paint_lightcone_surface_density(
                pixel_unit_vectors,
                catalog,
                concentration_params=ConcentrationParams(amplitude=amplitude),
            )
        )

    sigma = paint_lightcone_surface_density(pixel_unit_vectors, catalog)
    assert sigma.shape == (2,)
    assert jnp.all(jnp.isfinite(sigma))
    assert jnp.isfinite(jax.grad(objective)(5.71))


def test_lightcone_chunked_matches_unchunked():
    pixel_unit_vectors = jnp.array([[1.0, 0.0, 0.0], [0.999, 0.045, 0.0]])
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0], [0.998, 0.06, 0.0]]),
        chi=jnp.array([1000.0, 1050.0]),
        mass=jnp.array([1.0e14, 5.0e13]),
        redshift=jnp.array([0.3, 0.35]),
    )
    direct = paint_lightcone_surface_density(pixel_unit_vectors, catalog)
    chunked = paint_lightcone_surface_density(pixel_unit_vectors, catalog, chunk_size=1)
    assert jnp.allclose(direct, chunked, rtol=1.0e-5, atol=1.0e-5)


def test_lightcone_sparse_matches_dense_when_stencil_contains_all_pairs():
    pixel_unit_vectors = jnp.array(
        [[1.0, 0.0, 0.0], [0.999, 0.045, 0.0], [0.998, 0.06, 0.0]]
    )
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0], [0.998, 0.06, 0.0]]),
        chi=jnp.array([1000.0, 1050.0]),
        mass=jnp.array([1.0e14, 5.0e13]),
        redshift=jnp.array([0.3, 0.35]),
    )
    stencil = build_lightcone_sparse_stencil(pixel_unit_vectors, catalog, rmax_mpc_h=1.0e6)

    dense = paint_lightcone_surface_density(pixel_unit_vectors, catalog)
    sparse = paint_lightcone_surface_density_sparse(stencil, catalog)
    dense_mass = paint_lightcone_surface_density(
        pixel_unit_vectors,
        catalog,
        pixel_area_sr=0.01,
        return_mass_per_pixel=True,
    )
    sparse_mass = paint_lightcone_surface_density_sparse(
        stencil,
        catalog,
        pixel_area_sr=0.01,
        return_mass_per_pixel=True,
    )

    assert sparse.shape == (3,)
    assert stencil.size == 6
    assert jnp.allclose(sparse, dense, rtol=1.0e-5, atol=1.0e-5)
    assert jnp.allclose(sparse_mass, dense_mass, rtol=1.0e-5, atol=1.0e-5)


def test_lightcone_sparse_gradients_are_finite():
    pixel_unit_vectors = jnp.array([[1.0, 0.0, 0.0], [0.999, 0.045, 0.0]])
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
        chi=jnp.array([1000.0]),
        mass=jnp.array([1.0e14]),
        redshift=jnp.array([0.3]),
    )
    stencil = build_lightcone_sparse_stencil(pixel_unit_vectors, catalog, rmax_mpc_h=1.0e6)

    def concentration_objective(amplitude):
        return jnp.sum(
            paint_lightcone_surface_density_sparse(
                stencil,
                catalog,
                concentration_params=ConcentrationParams(amplitude=amplitude),
            )
        )

    def profile_objective(truncation_width_fraction):
        return jnp.sum(
            paint_lightcone_surface_density_sparse(
                stencil,
                catalog,
                profile_params=NFWProfileParams(
                    truncation_width_fraction=truncation_width_fraction
                ),
            )
        )

    assert jnp.isfinite(jax.grad(concentration_objective)(5.71))
    assert jnp.isfinite(jax.grad(profile_objective)(0.05))


def test_lightcone_sparse_builder_filters_pairs_and_handles_empty_stencils():
    pixel_unit_vectors = jnp.array(
        [[1.0, 0.0, 0.0], [0.999, 0.045, 0.0], [0.0, 1.0, 0.0]]
    )
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        chi=jnp.array([1000.0, 1000.0]),
        mass=jnp.array([1.0e14, 5.0e13]),
        redshift=jnp.array([0.3, 0.35]),
    )

    scalar = build_lightcone_sparse_stencil(pixel_unit_vectors, catalog, rmax_mpc_h=50.0)
    per_halo = build_lightcone_sparse_stencil(
        pixel_unit_vectors,
        catalog,
        rmax_mpc_h=jnp.array([50.0, 0.0]),
    )
    zero_rmax = build_lightcone_sparse_stencil(pixel_unit_vectors, catalog, rmax_mpc_h=0.0)
    empty = build_lightcone_sparse_stencil(
        jnp.array([[0.0, 0.0, 1.0]]),
        LightconeHaloCatalog(
            unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
            chi=jnp.array([1000.0]),
            mass=jnp.array([1.0e14]),
            redshift=jnp.array([0.3]),
        ),
        rmax_mpc_h=0.0,
    )
    painted_empty = paint_lightcone_surface_density_sparse(empty, catalog)

    assert scalar.pix_id.tolist() == [0, 1, 2]
    assert scalar.halo_id.tolist() == [0, 0, 1]
    assert per_halo.pix_id.tolist() == [0, 1, 2]
    assert per_halo.halo_id.tolist() == [0, 0, 1]
    assert zero_rmax.pix_id.tolist() == [0, 2]
    assert zero_rmax.halo_id.tolist() == [0, 1]
    assert empty.size == 0
    assert painted_empty.shape == (1,)
    assert jnp.all(jnp.isfinite(painted_empty))


def test_lightcone_particle_count_map_shape_grad_and_normalization():
    pixel_unit_vectors = jnp.array([[1.0, 0.0, 0.0], [0.999, 0.045, 0.0]])
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0]]),
        chi=jnp.array([1000.0]),
        mass=jnp.array([1.0e14]),
        redshift=jnp.array([0.3]),
    )
    particle_mass_msun_h = 1.0e10
    pixel_area_sr = 0.01

    def objective(amplitude):
        return jnp.sum(
            paint_lightcone_particle_count_map(
                pixel_unit_vectors,
                catalog,
                particle_mass_msun_h=particle_mass_msun_h,
                pixel_area_sr=pixel_area_sr,
                concentration_params=ConcentrationParams(amplitude=amplitude),
            )
        )

    mass_per_pixel = paint_lightcone_surface_density(
        pixel_unit_vectors,
        catalog,
        pixel_area_sr=pixel_area_sr,
        return_mass_per_pixel=True,
    )
    counts = paint_lightcone_particle_count_map(
        pixel_unit_vectors,
        catalog,
        particle_mass_msun_h=particle_mass_msun_h,
        pixel_area_sr=pixel_area_sr,
    )

    assert counts.shape == (2,)
    assert jnp.all(jnp.isfinite(counts))
    assert jnp.all(counts >= 0.0)
    assert jnp.allclose(counts, mass_per_pixel / particle_mass_msun_h, rtol=1.0e-6)
    assert jnp.isfinite(jax.grad(objective)(5.71))


def test_lightcone_particle_count_map_chunked_matches_unchunked():
    pixel_unit_vectors = jnp.array([[1.0, 0.0, 0.0], [0.999, 0.045, 0.0]])
    pixel_unit_vectors = pixel_unit_vectors / jnp.linalg.norm(pixel_unit_vectors, axis=1)[:, None]
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1.0, 0.0, 0.0], [0.998, 0.06, 0.0]]),
        chi=jnp.array([1000.0, 1050.0]),
        mass=jnp.array([1.0e14, 5.0e13]),
        redshift=jnp.array([0.3, 0.35]),
    )

    direct = paint_lightcone_particle_count_map(
        pixel_unit_vectors,
        catalog,
        particle_mass_msun_h=1.0e10,
        pixel_area_sr=0.01,
    )
    chunked = paint_lightcone_particle_count_map(
        pixel_unit_vectors,
        catalog,
        particle_mass_msun_h=1.0e10,
        pixel_area_sr=0.01,
        chunk_size=1,
    )

    assert jnp.allclose(direct, chunked, rtol=1.0e-5, atol=1.0e-5)
