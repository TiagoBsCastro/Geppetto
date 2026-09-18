import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.io import build_angular_histogram_geometry
from geppetto.painting_theory import (
    HistogramAngularGeometry,
    angular_assignment_moments,
    histogram_angular_assignment_moments,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _geometry():
    # Two equally weighted two-pixel haloes in group 0; pure NGP in group 1.
    return HistogramAngularGeometry(
        cosine_grid=jnp.cos(jnp.array([0., .2, .4])),
        analytic_ngp_weight=jnp.array([0., 1.]),
        response_lower_bin=jnp.array([0, 1, 0, 1]),
        response_bin_fraction=jnp.zeros(4), response_average_weight=jnp.full(4, .5),
        pair_row_a=jnp.array([0, 0, 1, 1, 2, 2, 3, 3]),
        pair_row_b=jnp.array([0, 1, 0, 1, 2, 3, 2, 3]),
        pair_lower_bin=jnp.array([0, 1, 1, 0, 0, 1, 1, 0]),
        pair_bin_fraction=jnp.zeros(8), pair_average_weight=jnp.full(8, .5),
    )


@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_population_histogram_matches_exact_moments_at_grid_nodes(chunk):
    geometry = _geometry()
    vectors = jnp.array([[0., 0., 1.], [np.sin(.2), 0., np.cos(.2)]])
    weights = jnp.array([.3, .7, .8, .2])
    result = jax.jit(lambda w: histogram_angular_assignment_moments(
        w, geometry, lmax=32, pair_chunk_size=chunk,
    ))(weights)
    first = angular_assignment_moments(vectors, weights[:2], vectors[0], lmax=32)
    second = angular_assignment_moments(vectors, weights[2:], vectors[0], lmax=32)
    for actual, one, two in zip(result, first, second, strict=True):
        assert actual.shape == (2, 33)
        np.testing.assert_allclose(actual[0], .5*(one+two), atol=2e-13)
        np.testing.assert_array_equal(actual[1], 1.)
    np.testing.assert_allclose(result.response[:, 0], 1., atol=1e-14)
    np.testing.assert_allclose(result.self_pair[:, 0], 1., atol=1e-14)


def test_population_histogram_jvp_includes_pair_products():
    geometry = _geometry()

    def evaluate(parameter):
        weight = jnp.array([parameter, 1-parameter, .8, .2])
        result = histogram_angular_assignment_moments(weight, geometry, lmax=16, pair_chunk_size=3)
        return jnp.stack(result)

    actual = jax.jacfwd(evaluate)(.3)
    expected = (evaluate(.30001) - evaluate(.29999)) / .00002
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)
    assert np.all(np.isfinite(actual))
    assert abs(actual[0, 0, -1]) > .01
    assert abs(actual[1, 0, -1]) > .01
    np.testing.assert_array_equal(actual[:, :, 0], 0.)
    np.testing.assert_array_equal(actual[:, 1], 0.)


def test_interpolated_angles_preserve_zero_mode_and_jvp():
    geometry = _geometry()._replace(
        response_bin_fraction=jnp.array([0., .25, 0., .25]),
        pair_bin_fraction=jnp.array([0., .25, .25, 0., 0., .25, .25, 0.]),
    )
    weights = jnp.array([.3, .7, .8, .2])
    result = histogram_angular_assignment_moments(weights, geometry, lmax=3)
    np.testing.assert_allclose(np.array(result)[:, :, 0], 1., atol=1e-14)
    derivative = jax.jacfwd(lambda x: histogram_angular_assignment_moments(
        jnp.array([x, 1-x, .8, .2]), geometry, lmax=3,
    ).self_pair)(.3)
    np.testing.assert_allclose(derivative[:, 0], 0., atol=4*np.finfo(np.float64).eps)


def test_analytic_ngp_without_native_rows_or_pairs():
    empty_int = jnp.empty(0, dtype=int)
    empty_float = jnp.empty(0)
    geometry = HistogramAngularGeometry(
        jnp.array([1., .9]), jnp.array([1., 0.]), empty_int, empty_float, empty_float,
        empty_int, empty_int, empty_int, empty_float, empty_float,
    )
    result = histogram_angular_assignment_moments(empty_float, geometry, lmax=5)
    for value in result:
        np.testing.assert_array_equal(value[0], 1.)
        np.testing.assert_array_equal(value[1], 0.)


def test_population_geometry_shape_validation():
    with pytest.raises(ValueError, match="response geometry"):
        histogram_angular_assignment_moments(jnp.ones(3), _geometry(), lmax=1)
    with pytest.raises(ValueError, match="pair geometry"):
        histogram_angular_assignment_moments(
            jnp.ones(4), _geometry()._replace(pair_row_b=jnp.zeros(2, dtype=int)), lmax=1,
        )


def test_float32_weights_with_float64_geometry_promote_without_scatter_downcasts():
    result = histogram_angular_assignment_moments(
        jnp.array([.25, .75, .5, .5], dtype=jnp.float32), _geometry(), lmax=8,
    )
    assert result.response.dtype == jnp.float64
    assert result.self_pair.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(result)[:, :, 0], 1., atol=1e-14)


def _host_geometry_arguments():
    pair = np.array([[0., 0., 1.], [np.sin(.2), 0., np.cos(.2)]])
    return dict(
        native_pixel_unit_vectors=np.tile(pair, (2, 1)),
        halo_row_offsets=np.array([0, 2, 4]),
        halo_reference_unit_vectors=np.tile(pair[0], (2, 1)),
        halo_group_indices=np.array([0, 0]),
        halo_average_weights=np.array([.5, .5]),
        analytic_ngp_weights=np.array([0., 1.]),
        angle_grid_rad=np.array([0., .2, .4]),
    )


@pytest.mark.parametrize("chunk", [1, 3])
def test_host_geometry_matches_manual_same_halo_pairs(chunk):
    geometry = build_angular_histogram_geometry(**_host_geometry_arguments(), pair_chunk_size=chunk)
    weights = jnp.array([.3, .7, .8, .2])
    result = histogram_angular_assignment_moments(weights, geometry, lmax=32)
    expected = histogram_angular_assignment_moments(weights, _geometry(), lmax=32)
    np.testing.assert_allclose(result, expected, atol=1.e-13)
    assert geometry.pair_row_a.size == 8
    np.testing.assert_array_equal(geometry.pair_row_a//2, geometry.pair_row_b//2)


def test_empty_host_geometry_can_supply_pure_ngp_groups():
    geometry = build_angular_histogram_geometry(
        np.empty((0, 3)), np.array([0]), np.empty((0, 3)), np.empty(0, dtype=int),
        np.empty(0), np.array([1.]), np.array([0., .1]),
    )
    result = histogram_angular_assignment_moments(jnp.empty(0), geometry, lmax=3)
    np.testing.assert_array_equal(result, 1.)


@pytest.mark.parametrize("change, message", [
    ({"halo_row_offsets": np.array([0, 2, 5])}, "partition native rows"),
    ({"halo_group_indices": np.array([0, 2])}, "valid integer group"),
    ({"halo_average_weights": np.array([.5, -.5])}, "non-negative halo"),
    ({"analytic_ngp_weights": np.array([0., np.nan])}, "non-negative group"),
    ({"angle_grid_rad": np.array([0., .1])}, "does not cover"),
    ({"angle_grid_rad": np.array([0., 0.])}, "increase strictly"),
    ({"halo_reference_unit_vectors": np.zeros((2, 3))}, "finite unit vectors"),
])
def test_host_geometry_rejects_invalid_domain(change, message):
    arguments = _host_geometry_arguments() | change
    with pytest.raises(ValueError, match=message):
        build_angular_histogram_geometry(**arguments)
