from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.io import (
    PinocchioCatalogError,
    PinocchioMassFunction,
    pinocchio_halo_count_quadrature_from_tables,
    read_pinocchio_halo_count_quadrature,
)
from geppetto.theory import HaloCountQuadrature, halo_count_weights


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _table(redshift=0., mass=(8.e12, 1.e12, 3.e12), counts=(2, 10, 0)):
    mass = np.asarray(mass)
    zeros = np.zeros_like(mass)
    return PinocchioMassFunction(
        mass_msun_h=mass,
        number_density=zeros,
        number_density_plus_1sigma=zeros,
        number_density_minus_1sigma=zeros,
        halo_counts=np.asarray(counts),
        analytic_number_density=zeros,
        peak_height_nu=np.ones_like(mass),
        source=Path("synthetic.mf.out"),
        redshift=redshift,
    )


def test_native_count_and_representative_mass_moments_close_without_mass_interpolation():
    tables = [_table(), _table(1., (2.e12, 9.e12), (7, 1))]
    result = pinocchio_halo_count_quadrature_from_tables(tables, box_size_mpc_h=100.)
    np.testing.assert_allclose(result.scale_factor, [.5, 1.])
    mass = np.exp(result.log_mass_msun_h)
    np.testing.assert_allclose(mass, [1.e12, 2.e12, 3.e12, 8.e12, 9.e12])
    for table in tables:
        weights = np.asarray(halo_count_weights(table.redshift, result))
        assert np.count_nonzero(weights) == np.count_nonzero(table.halo_counts)
        for order in (0, 1, 2):
            expected = np.sum(table.halo_counts * table.mass_msun_h**order) / 100.**3
            np.testing.assert_allclose(np.sum(weights * mass**order), expected, rtol=1.e-14)
    # Deliberately zero density columns cannot affect count-based quadrature.
    assert float(jnp.sum(result.number_density_weight_mpc_h3)) > 0.


def test_scale_factor_interpolation_preserves_moments_and_shapes():
    quadrature = pinocchio_halo_count_quadrature_from_tables(
        [_table(), _table(1., (2.e12, 9.e12), (7, 1))], box_size_mpc_h=100.,
    )
    result = jax.jit(halo_count_weights)(jnp.array([[1., 1./.75-1., 0.]]), quadrature)
    assert result.shape == (1, 3, 5)
    np.testing.assert_allclose(result[0, 1], jnp.mean(quadrature.number_density_weight_mpc_h3, axis=0))
    np.testing.assert_allclose(halo_count_weights(10., quadrature), result[0, 0])
    np.testing.assert_allclose(halo_count_weights(-.1, quadrature), result[0, 2])


def test_count_quadrature_is_a_differentiable_pytree():
    table = pinocchio_halo_count_quadrature_from_tables(
        [_table(), _table(1., (2.e12, 9.e12), (7, 1))], box_size_mpc_h=100.,
    )
    mass = jnp.exp(table.log_mass_msun_h) / 1.e12

    def integral(weights, z):
        return jnp.sum(halo_count_weights(z, table._replace(
            number_density_weight_mpc_h3=weights,
        )) * mass**2)

    gradient = jax.jit(jax.grad(integral))(table.number_density_weight_mpc_h3, 1./.75-1.)
    np.testing.assert_allclose(gradient, jnp.broadcast_to(.5*mass**2, gradient.shape))
    dz = jax.grad(integral, argnums=1)(table.number_density_weight_mpc_h3, .3)
    step = 1.e-5
    difference = (integral(table.number_density_weight_mpc_h3, .3+step)
                  - integral(table.number_density_weight_mpc_h3, .3-step)) / (2*step)
    assert np.isfinite(dz) and dz != 0.
    np.testing.assert_allclose(dz, difference, rtol=1.e-8)


def test_single_snapshot_and_zero_count_bins():
    result = pinocchio_halo_count_quadrature_from_tables([_table()], box_size_mpc_h=10.)
    assert isinstance(result, HaloCountQuadrature)
    np.testing.assert_allclose(halo_count_weights(100., result), [.01, 0., .002])
    assert halo_count_weights(jnp.array([0., 1.]), result).shape == (2, 3)


def test_count_reader_uses_native_counts_and_box_volume(tmp_path):
    path = tmp_path / "pinocchio.0.0000.000.mf.out"
    path.write_text("# Mass function for redshift 0.0\n1e12 1e-17 1e-17 0 9 1e-17 1\n")
    result = read_pinocchio_halo_count_quadrature([path], box_size_mpc_h=10.)
    np.testing.assert_allclose(result.number_density_weight_mpc_h3, [[.009]])


@pytest.mark.parametrize("box_size", [0., -1., np.nan, np.inf, 1.e200, 1.e-200])
def test_invalid_box_volume_fails(box_size):
    with pytest.raises(PinocchioCatalogError, match="finite and positive"):
        pinocchio_halo_count_quadrature_from_tables([_table()], box_size_mpc_h=box_size)


@pytest.mark.parametrize("change, message", [
    ({"redshift": None}, "redshift in every"),
    ({"redshift": np.nan}, "redshifts must be finite"),
    ({"redshift": -1.}, "redshifts must be finite"),
    ({"mass_msun_h": np.array([1., 1., 2.])}, "unique 1D grid"),
    ({"mass_msun_h": np.array([1., np.inf, 2.])}, "unique 1D grid"),
    ({"halo_counts": np.array([1, -1, 0])}, "non-negative integers"),
    ({"halo_counts": np.array([1, .5, 0])}, "non-negative integers"),
    ({"halo_counts": np.array([1, np.nan, 0])}, "non-negative integers"),
    ({"halo_counts": np.array([1, 0])}, "non-negative integers"),
])
def test_invalid_native_table_fails(change, message):
    with pytest.raises(PinocchioCatalogError, match=message):
        pinocchio_halo_count_quadrature_from_tables(
            [replace(_table(), **change)], box_size_mpc_h=100.,
        )


def test_empty_or_duplicate_snapshots_fail():
    with pytest.raises(PinocchioCatalogError, match="at least one"):
        pinocchio_halo_count_quadrature_from_tables([], box_size_mpc_h=100.)
    with pytest.raises(PinocchioCatalogError, match="redshifts must be unique"):
        pinocchio_halo_count_quadrature_from_tables([_table(), _table()], box_size_mpc_h=100.)


def test_overflowing_number_density_fails_even_with_a_finite_box_volume():
    with pytest.raises(PinocchioCatalogError, match="number-density weights must be finite"):
        pinocchio_halo_count_quadrature_from_tables([_table()], box_size_mpc_h=1.e-105)
