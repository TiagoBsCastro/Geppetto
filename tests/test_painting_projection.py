import argparse
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.painting_theory import (
    StationaryLineOfSightRule,
    project_constrained_painting_node,
    resolved_painting_population,
    stationary_shell_average,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("project_painting_covariance")


def _population():
    return resolved_painting_population(
        jnp.array([2., 8.]), jnp.array([.02, .005]), jnp.array([1.5, 3.]), 1., .1,
    )


def test_population_keeps_resolved_mass_and_bias_integrals_below_unity():
    population = jax.jit(resolved_painting_population)(
        jnp.array([2., 8.]), jnp.array([.02, .005]), jnp.array([1.5, 3.]), 1., .1,
    )
    np.testing.assert_allclose(population.halo_mass_fraction, [.04, .04])
    np.testing.assert_allclose(population.halo_bias_weight, [.06, .12])
    np.testing.assert_allclose(population.halo_self_power, [.08, .32])
    assert float(population.uncollapsed_mass_fraction) == pytest.approx(.92)
    assert float(population.uncollapsed_self_power) == pytest.approx(.092)
    assert 1-float(population.halo_bias_weight.sum()) == pytest.approx(.82)


def test_population_empty_catalogue_and_mass_weight_derivatives():
    empty = resolved_painting_population(jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), 2., 3.)
    assert empty.halo_self_power.shape == (0,)
    assert empty.uncollapsed_self_power == 1.5
    derivative = jax.jacfwd(lambda mass: resolved_painting_population(
        mass, jnp.array([.02, .005]), jnp.ones(2), 2., .1,
    ).halo_self_power)(jnp.array([2., 8.]))
    np.testing.assert_allclose(derivative, np.diag([.02, .02]))


def test_population_rejects_wrong_shapes_without_host_callbacks():
    with pytest.raises(ValueError, match="n_mass"):
        resolved_painting_population(jnp.ones(2), jnp.ones(3), jnp.ones(2), 1., 1.)
    with pytest.raises(ValueError, match="scalar"):
        resolved_painting_population(jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.ones(1), 1.)


def test_projection_has_linear_and_discrete_self_limits_without_extra_noise():
    population = _population()
    result = jax.jit(project_constrained_painting_node)(
        population, jnp.array([10., 0.]), jnp.array([1., 0.]), .2,
        jnp.array([1., .4]), jnp.ones((2, 2)), jnp.ones((2, 2)),
    )
    assert np.asarray(result).shape == (5, 2)
    np.testing.assert_allclose(result.total, [2., .2*(.092+.08+.32)], rtol=2.e-14)
    np.testing.assert_allclose(result.one_halo, .2*(.08+.32), rtol=2.e-14)
    np.testing.assert_allclose(np.sum(np.asarray(result)[:4], axis=0), result.total, rtol=2.e-14)


def test_projected_profile_self_is_D_not_A_squared_or_windowed():
    population = _population()
    response = jnp.array([[.2, .4]])
    self_pair = jnp.array([[.3, .5]])
    result = project_constrained_painting_node(
        population, jnp.zeros(1), jnp.zeros(1), 3., jnp.array([.1]), response, self_pair,
    )
    assert float(result.one_halo[0]) == pytest.approx(3*(.08*.3+.32*.5))
    assert float(result.total[0]) == pytest.approx(3*(.092+.08*.3+.32*.5))


def test_projected_derivative_includes_response_cross_terms_and_self_pairs():
    def evaluate(parameter):
        response = jnp.array([[1., 1.], [.4+parameter, .7-.1*parameter]])
        self_pair = jnp.array([[1., 1.], [.7+.2*parameter, .8-.2*parameter]])
        return jnp.stack(project_constrained_painting_node(
            _population(), jnp.array([10., 2.]), jnp.array([1., .5]), .2,
            jnp.array([1., .9]), response, self_pair,
        ))
    derivative = jax.jit(jax.jacfwd(evaluate))(.1)
    finite = (evaluate(.10001)-evaluate(.09999))/.00002
    assert derivative.shape == (5, 2)
    np.testing.assert_allclose(derivative, finite, rtol=1.e-8, atol=1.e-11)
    np.testing.assert_array_equal(derivative[:, 0], 0.)
    assert abs(float(derivative[1, 1])) > .01
    assert abs(float(derivative[3, 1])) > .001
    np.testing.assert_allclose(derivative[:4].sum(axis=0), derivative[-1], atol=1.e-14)


def test_stationary_projection_preserves_constants_and_exposes_continuations(example):
    rule = example.line_of_sight_rule(32, 16)
    result = jax.jit(stationary_shell_average)(
        jnp.array([.001, 1., 10.]), jnp.array([.1, 2.]), jnp.array([7., 7.]), 2., rule,
        low_k_value=7., high_k_value=7.,
    )
    assert result.value.shape == (3,)
    np.testing.assert_allclose(result.value, 7., rtol=2.e-14)
    assert result.low_k_continuation[0] > 0
    assert result.low_k_continuation[1] == 0
    np.testing.assert_allclose(result.high_k_continuation[-1], 7., rtol=2.e-14)


def test_stationary_projection_matches_direct_discrete_rule_and_gradients():
    rule = StationaryLineOfSightRule(jnp.array([.1, 1., 3.]), jnp.array([.6, .3, .05]), .05)
    k = np.array([.5, 3.])
    wave, power = np.array([.2, 1., 2.]), np.array([2., 3., 4.])
    values = np.sqrt(k[:, None]**2+np.asarray(rule.nodes)[None, :]**2)
    expected = np.interp(values, wave, power, left=1., right=5.) @ np.array(rule.weights)+.05*5
    def evaluate(table):
        return stationary_shell_average(k, wave, table, 2., rule, low_k_value=1., high_k_value=5.).value
    np.testing.assert_allclose(evaluate(power), expected)
    actual = jax.jit(jax.jacfwd(evaluate))(jnp.array(power))
    finite = np.stack([(evaluate(power+np.eye(3)[i]*1.e-5)
                       - evaluate(power-np.eye(3)[i]*1.e-5))/2.e-5 for i in range(3)], axis=-1)
    np.testing.assert_allclose(actual, finite, rtol=1.e-9, atol=1.e-10)
    assert np.all(np.isfinite(actual))


def test_stationary_projection_validates_shape_and_quadrature(example):
    rule = example.line_of_sight_rule(4, 16)
    with pytest.raises(ValueError, match="matching nodes"):
        stationary_shell_average(jnp.ones(2), jnp.ones(1), jnp.ones(1), 1., rule,
                                 low_k_value=1., high_k_value=1.)
    with pytest.raises(ValueError, match="scalars"):
        stationary_shell_average(jnp.ones(2), jnp.array([1., 2.]), jnp.ones(2), jnp.ones(2), rule,
                                 low_k_value=1., high_k_value=1.)
    with pytest.raises(ValueError, match="integer"):
        example.line_of_sight_rule(1.5, 16)
    with pytest.raises(ValueError, match="integer"):
        example.line_of_sight_rule(4, 2)


def _fixture(tmp_path, *, gradients=False, linear_reference=True):
    data = dict(
        mass_msun_h=np.array([2., 8.]), number_density_weight_mpc_h3=np.array([.02, .005]),
        linear_halo_bias=np.array([1.5, 3.]), k_h_mpc=np.array([1.e-4, 10.]),
        coherent_power_mpc_h3=np.array([10., 10.]), linear_power_mpc_h3=np.array([8., 8.]),
        same_protohalo_power_mpc_h3=np.array([.2, .2]), lattice_correction_mpc_h3=np.zeros(2),
        response=np.full((4, 2), .9), self_pair=np.full((4, 2), .95),
    )
    if gradients:
        data.update(response_jacobian=np.full((3, 4, 2), -.01),
                    self_pair_jacobian=np.full((3, 4, 2), -.005))
    np.savez(tmp_path / "node.npz", **data)
    grid = dict(ell=np.arange(2, 6), pixel_window=np.array([1., .9, .8, .7]))
    if linear_reference:
        grid.update(segment_indices=np.array([4]), shell_linear=np.full((1, 4), 1.e-5))
    np.savez(tmp_path / "grid.npz", **grid)
    specification = dict(
        schema_version=1, mean_density_msun_h_mpch3=1., particle_mass_msun_h=.1, grid="grid.npz",
        backbone_description="synthetic algebraic covariance fixture, not cosmological theory",
        assignment_description="synthetic normalized moments, not a painted catalogue",
        nodes=[dict(segment_index=4, redshift=.2, chi_lo_mpc_h=500., chi_hi_mpc_h=600.,
                    chi_mpc_h=chi, dchi_weight_mpc_h=50., table="node.npz") for chi in (520., 580.)],
    )
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(specification))
    args = argparse.Namespace(inputs=path, output=tmp_path / "prediction.npz", projection="stationary",
                              los_periods=4, los_order=16)
    return args, data, specification


@pytest.mark.parametrize("projection", ["limber", "stationary"])
@pytest.mark.parametrize("gradients", [False, True])
def test_portable_projection_replays_all_components_and_linear_replacement(tmp_path, example, projection, gradients):
    args, _, _ = _fixture(tmp_path, gradients=gradients)
    args.projection = projection
    report = example.run(args)
    assert report["linear_projection_replaced"]
    assert len(report["input_sha256"]) == 2
    with np.load(args.output, allow_pickle=False) as result:
        components = result["stationary_components"]
        assert components.shape == (1, 5, 4)
        np.testing.assert_allclose(components[:, :4].sum(axis=1), components[:, -1], atol=1.e-18)
        np.testing.assert_allclose(result["shell_total"], components[:, -1]+result["linear_projection_correction"])
        np.testing.assert_allclose(result["stationary_linear"]+result["linear_projection_correction"], 1.e-5)
        assert ("component_jacobian" in result) == gradients
        if gradients:
            jacobian = result["component_jacobian"]
            assert jacobian.shape == (1, 3, 5, 4)
            assert np.all(np.isfinite(jacobian))
            np.testing.assert_allclose(jacobian[:, :, :4].sum(axis=2), jacobian[:, :, -1], atol=1.e-18)
        assert json.loads(result["metadata_json"].item())["projection"] == projection


def test_no_linear_reference_leaves_stationary_prediction_unchanged(tmp_path, example):
    args, _, _ = _fixture(tmp_path, linear_reference=False)
    example.run(args)
    with np.load(args.output) as result:
        np.testing.assert_array_equal(result["linear_projection_correction"], 0.)
        np.testing.assert_array_equal(result["shell_total"], result["stationary_components"][:, -1])


def test_fits_derived_big_endian_arrays_are_converted_before_jax(tmp_path, example):
    args, data, _ = _fixture(tmp_path)
    np.savez(tmp_path / "node.npz", **{key: value.astype(">f8") for key, value in data.items()})
    np.savez(tmp_path / "grid.npz", ell=np.arange(2, 6).astype(">i8"),
             pixel_window=np.ones(4, dtype=">f8"))
    loaded = example._load(tmp_path / "node.npz")
    assert all(value.dtype.isnative for value in loaded.values())
    example.run(args)
    with np.load(args.output) as result:
        assert np.all(np.isfinite(result["shell_total"]))


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data.update(number_density_weight_mpc_h3=np.array([1., 1.])), "mass fraction"),
    (lambda data: data.update(same_protohalo_power_mpc_h3=np.full(2, 20.)), "constraint"),
    (lambda data: data.update(response=np.full((4, 2), 1.1)), "global assignment"),
    (lambda data: data.update(coherent_power_mpc_h3=np.array([np.nan, 1.])), "finite"),
    (lambda data: data.update(response_jacobian=np.zeros((3, 4, 2))), "together"),
    (lambda data: data.update(k_h_mpc=np.array([1., 1.])), "k grid"),
])
def test_portable_projection_rejects_invalid_physical_inputs(tmp_path, example, mutation, match):
    args, data, _ = _fixture(tmp_path)
    mutation(data)
    np.savez(tmp_path / "node.npz", **data)
    with pytest.raises(ValueError, match=match):
        example.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda spec: spec["nodes"].append(spec["nodes"][0]), "duplicate"),
    (lambda spec: spec["nodes"][0].update(dchi_weight_mpc_h=51.), "shell width"),
    (lambda spec: spec["nodes"][0].update(chi_mpc_h=499.), "inside"),
    (lambda spec: spec.update(schema_version=9), "schema_version"),
    (lambda spec: spec.update(assignment_description=""), "assignment_description"),
])
def test_portable_projection_rejects_inconsistent_node_specifications(tmp_path, example, mutation, match):
    args, _, spec = _fixture(tmp_path)
    mutation(spec)
    args.inputs.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        example.run(args)
    assert not args.output.exists()


def test_projection_will_not_overwrite_input(tmp_path, example):
    args, _, _ = _fixture(tmp_path)
    args.output = tmp_path / "grid.npz"
    original = args.output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        example.run(args)
    assert args.output.read_bytes() == original


def test_output_suffix_cannot_bypass_input_overwrite_protection(tmp_path, example):
    args, _, _ = _fixture(tmp_path)
    args.output = tmp_path / "grid"
    with pytest.raises(ValueError, match="suffix"):
        example.run(args)
