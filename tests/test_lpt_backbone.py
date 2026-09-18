from dataclasses import replace
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.io import PinocchioCatalogError, read_pinocchio_lpt_growth_ratios
from geppetto.lpt_backbone import (
    LPTBackboneParams,
    LPTGrowthRatios,
    build_lpt_particle_backbone,
    cubic_lattice_shells,
    spherical_protohalo_overlap,
)
from geppetto.painting_theory import project_constrained_painting_node, resolved_painting_population


@pytest.mark.parametrize("radius", [0., .9, 1., 2., 3.5])
def test_lattice_shell_multiplicity_matches_direct_integer_enumeration(radius):
    distance, multiplicity = cubic_lattice_shells(1., radius)
    n = int(radius)
    points = np.array(list(product(range(-n, n+1), repeat=3)))
    exact = np.linalg.norm(points, axis=1)
    expected_distance, expected_count = np.unique(exact[exact <= radius], return_counts=True)
    np.testing.assert_array_equal(distance, expected_distance)
    np.testing.assert_array_equal(multiplicity, expected_count)
    assert multiplicity[0] == 1


def test_spherical_overlap_boundaries_and_scaling():
    np.testing.assert_allclose(spherical_protohalo_overlap(np.array([0., 1., 2., 3.]), 1.),
                               [1., 5/16, 0., 0.])
    np.testing.assert_array_equal(spherical_protohalo_overlap(np.array([0., 5., 10.]), 5.),
                                  spherical_protohalo_overlap(np.array([0., 1., 2.]), 1.))
    with pytest.raises(ValueError, match="R > 0"):
        spherical_protohalo_overlap(np.array([0.]), 0.)


def _growth_file(path, second=1., third=1.):
    data = np.ones((3, 20))
    data[:, 0] = [.25, .5, 1.]
    data[:, 6] = [.1, .4, 1.]
    data[:, 7] = (3/7)*data[:, 6]**2*second
    data[:, 9] = (5/42)*data[:, 6]**3*third
    np.savetxt(path, data)
    return data


@pytest.mark.parametrize("ratios", [(1., 1.), (1.02, 1.04)])
def test_growth_reader_forms_ratios_before_interpolation(tmp_path, ratios):
    path = tmp_path / "cosmology.out"
    _growth_file(path, *ratios)
    np.testing.assert_allclose(read_pinocchio_lpt_growth_ratios(path, 1.5), ratios, rtol=1.e-14)
    # Interpolating D2/D3 independently and only then dividing would fail here.


def test_growth_reader_rejects_extrapolation_and_invalid_entries(tmp_path):
    path = tmp_path / "cosmology.out"
    data = _growth_file(path)
    with pytest.raises(PinocchioCatalogError, match="outside"):
        read_pinocchio_lpt_growth_ratios(path, 10.)
    with pytest.raises(PinocchioCatalogError, match="nonnegative"):
        read_pinocchio_lpt_growth_ratios(path, -1.)
    data[1, 9] = 0.
    np.savetxt(path, data)
    with pytest.raises(PinocchioCatalogError, match="invalid"):
        read_pinocchio_lpt_growth_ratios(path, 0.)


def _arguments():
    wave = np.geomspace(1.e-4, 10., 512)
    return dict(
        linear_k_h_mpc=wave, linear_power_mpc_h3=1.e3*wave/(1+(wave/.1)**3),
        mass_msun_h=np.array([10., 100.]), number_density_weight_mpc_h3=np.array([.002, .0004]),
        mean_density_msun_h_mpch3=1., particle_mass_msun_h=1., grid_spacing_mpc_h=1.,
        growth=LPTGrowthRatios(1., 1.),
        params=LPTBackboneParams(k_min_h_mpc=.01, k_max_h_mpc=1., k_nodes=6,
                                 q_max_mpc_h=16., continuous_order=128, spectral_order=128,
                                 angular_order=64, fft_size=1024, bessel_orders=12,
                                 lens_order=8, conditioning_radial_nodes=65),
    )


@pytest.fixture(scope="module")
def small_backbone():
    pytest.importorskip("velocileptors")
    return build_lpt_particle_backbone(**_arguments())


def test_backbone_shape_mass_budget_and_zero_mode(small_backbone):
    for array in (small_backbone.k_h_mpc, small_backbone.coherent_power, small_backbone.linear_power,
                  small_backbone.lattice_correction, small_backbone.same_protohalo_power):
        assert array.shape == (6,)
        assert np.all(np.isfinite(array))
    assert small_backbone.resolved_mass_fraction == pytest.approx(.06)
    assert small_backbone.halo_self_power == pytest.approx(4.2)
    assert small_backbone.same_protohalo_zero_mode == pytest.approx(4.2, rel=3.e-12)
    assert np.all(small_backbone.coherent_power >= 0)


def test_host_backbone_still_allows_differentiable_assignment(small_backbone):
    # Orthogonal pixels each with reference cosine 1/2 give A=1/2 and
    # D=p^2+(1-p)^2 at ell=1. The backbone remains fixed.
    population = resolved_painting_population(jnp.array([10., 100.]), jnp.array([.002, .0004]),
                                              jnp.array([1.5, 2.]), 1., 1.)
    def evaluate(parameter):
        response = jnp.full((1, 2), .5)
        self_pair = jnp.full((1, 2), parameter**2+(1-parameter)**2)
        return project_constrained_painting_node(
            population, jnp.asarray(small_backbone.coherent_power[:1]), jnp.zeros(1),
            .1, jnp.ones(1), response, self_pair,
        ).total[0]
    derivative = jax.jit(jax.grad(evaluate))(.3)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, .1*4.2*(4*.3-2), rtol=2.e-6)


def test_backbone_chunk_size_does_not_change_the_prediction(small_backbone):
    pytest.importorskip("velocileptors")
    arguments = _arguments()
    arguments["params"] = replace(arguments["params"], k_chunk_size=1)
    result = build_lpt_particle_backbone(**arguments)
    np.testing.assert_allclose(result.coherent_power, small_backbone.coherent_power, rtol=1.e-12)
    np.testing.assert_allclose(result.same_protohalo_power, small_backbone.same_protohalo_power, rtol=1.e-12)
    np.testing.assert_allclose(result.lattice_correction, small_backbone.lattice_correction, atol=1.e-12)


def test_empty_resolved_population_has_no_subtraction():
    pytest.importorskip("velocileptors")
    arguments = _arguments()
    arguments.update(mass_msun_h=np.empty(0), number_density_weight_mpc_h3=np.empty(0))
    result = build_lpt_particle_backbone(**arguments)
    np.testing.assert_array_equal(result.same_protohalo_power, 0.)
    assert result.same_protohalo_zero_mode == 0.
    assert result.resolved_mass_fraction == 0.


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a.update(particle_mass_msun_h=2.), "cell volume"),
    (lambda a: a.update(number_density_weight_mpc_h3=np.ones(2)), "mass fraction"),
    (lambda a: a.update(mass_msun_h=np.array([.5, 10.])), "one simulation particle"),
    (lambda a: a.update(growth=LPTGrowthRatios(-1., 1.)), "positive"),
    (lambda a: a.update(params=replace(a["params"], q_max_mpc_h=1.)), "diameter"),
    (lambda a: a.update(params=replace(a["params"], spectral_order=3)), "spectral_order"),
])
def test_backbone_rejects_inconsistent_inputs(mutation, match):
    arguments = _arguments()
    mutation(arguments)
    with pytest.raises(ValueError, match=match):
        build_lpt_particle_backbone(**arguments)
