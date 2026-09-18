"""Small geometry tests for the read-only, host-membership close-up plot."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
SPEC = importlib.util.spec_from_file_location(
    "galaxy_halo_plot", Path(__file__).parents[1]/"examples/plot_most_massive_galaxy_halo.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tangent_projection_origin_orientation_and_scale():
    centre = np.array([100., 0., 0.])
    positions = np.array([[100., 0., 0.], [100., 1., 0.], [100., 0., 2.]])
    result = MODULE.tangent_offsets(positions, centre, np.eye(3))
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, np.array([[0., 0.], [.01, 0.], [0., .02]])*MODULE.ARCMIN_PER_RADIAN)


def test_tangent_projection_respects_catalogue_basis():
    centre = np.array([0., 100., 0.])
    positions = np.array([[0., 100., 0.], [-1., 100., 0.]])
    basis = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    result = MODULE.tangent_offsets(positions, centre, basis)
    np.testing.assert_allclose(result, np.array([[0., 0.], [.01, 0.]])*MODULE.ARCMIN_PER_RADIAN)


def test_projected_sphere_boundary_is_the_tangent_silhouette():
    radius, distance = 2., 100.
    angle = np.arcsin(radius/distance)
    np.testing.assert_allclose(MODULE.support_radius_arcmin(radius, distance),
                               np.tan(angle)*MODULE.ARCMIN_PER_RADIAN)
    with pytest.raises(ValueError, match="support radius"):
        MODULE.support_radius_arcmin(100., 100.)
    with pytest.raises(ValueError, match="hemisphere"):
        MODULE.tangent_offsets(np.array([[-100., 0., 0.]]), np.array([100., 0., 0.]), np.eye(3))
