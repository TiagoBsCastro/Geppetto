import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_legendre

from geppetto.catalog import AdaptiveLightconeStencil, LightconeHaloCatalog
from geppetto.concentration import ConcentrationParams
from geppetto.painters import paint_lightcone_particle_count_map_sparse
from geppetto.painting_theory import (
    ConstrainedBackboneAngularModel,
    HaloBackboneAngularSpectra,
    angular_assignment_moments,
    assemble_constrained_painted_angular_power,
    assemble_painted_angular_power,
    catalogue_self_pair_amplitude,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("lmax", [0, 1, 64])
def test_ngp_has_unit_response_and_white_self_power(lmax):
    axis = jnp.array([0.0, 0.0, 1.0])
    result = jax.jit(lambda weights: angular_assignment_moments(
        axis[None, :], weights, axis, lmax=lmax
    ))(jnp.ones(1))
    assert result.response.shape == (lmax + 1,)
    np.testing.assert_allclose(result.response, 1, rtol=0, atol=0)
    np.testing.assert_allclose(result.self_pair, 1, rtol=0, atol=0)


@pytest.mark.parametrize("chunk", [1, 2, 17])
def test_two_pixel_moments_match_analytic_addition_theorem(chunk):
    theta, mass = 0.13, 0.3
    vectors = jnp.array([[np.sin(theta), 0, np.cos(theta)], [-np.sin(theta), 0, np.cos(theta)]])
    result = angular_assignment_moments(
        vectors, jnp.array([mass, 1 - mass]), jnp.array([0., 0., 1.]),
        lmax=32, pair_chunk_size=chunk,
    )
    ell = np.arange(33)
    np.testing.assert_allclose(result.response, eval_legendre(ell, np.cos(theta)), atol=1e-13)
    expected = mass**2 + (1 - mass)**2 + 2 * mass * (1 - mass) * eval_legendre(
        ell, np.cos(2 * theta)
    )
    np.testing.assert_allclose(result.self_pair, expected, atol=1e-13)
    assert np.all(np.asarray(result.self_pair - result.response**2) >= -1e-13)


def test_response_and_self_pair_cannot_be_interchanged():
    vectors = jnp.array([[0., 0., 1.], [1., 0., 0.]])
    moments = angular_assignment_moments(
        vectors, jnp.array([0.5, 0.5]), jnp.array([0., 0., 1.]), lmax=3
    )
    assert moments.response[1] ** 2 == pytest.approx(0.25)
    assert moments.self_pair[1] == pytest.approx(0.5)


def test_clipped_assignment_is_not_renormalized():
    result = angular_assignment_moments(
        jnp.array([[0., 0., 1.]]), jnp.array([0.4]), jnp.array([0., 0., 1.]), lmax=4
    )
    np.testing.assert_allclose(result.response, 0.4)
    np.testing.assert_allclose(result.self_pair, 0.16)


def test_padding_and_empty_assignment():
    vector = jnp.array([0., 0., 1.])
    padded = angular_assignment_moments(
        jnp.stack([vector, vector, vector]), jnp.array([1., 0., 0.]), vector,
        lmax=5, pair_chunk_size=2,
    )
    np.testing.assert_allclose(padded.self_pair, 1)
    empty = angular_assignment_moments(jnp.empty((0, 3)), jnp.empty(0), vector, lmax=5)
    np.testing.assert_array_equal(empty.response, np.zeros(6))
    np.testing.assert_array_equal(empty.self_pair, np.zeros(6))


@pytest.mark.parametrize("ngp", [False, True])
def test_actual_painter_concentration_jvp_includes_global_normalization(ngp):
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.array([[1., 0., 0.]]), chi=jnp.array([1000.]),
        mass=jnp.array([1.e14]), redshift=jnp.array([0.3]),
    )
    stencil = AdaptiveLightconeStencil(
        sample_compact_row=jnp.array([0, 1]), sample_halo_id=jnp.array([0, 0]),
        sample_r_perp=jnp.array([0.05, 0.15]), sample_solid_angle_sr=jnp.full(2, 1e-8),
        sample_valid=jnp.full(2, not ngp), sample_in_compact=jnp.ones(2, dtype=bool),
        ngp_compact_row=jnp.array([0]), ngp_active=jnp.array([ngp]),
        ngp_in_compact=jnp.array([ngp]), resolved_halo_mask=jnp.array([not ngp]), n_pix=2,
    )
    vectors = jnp.array([[1., 0., 0.], [np.cos(.01), np.sin(.01), 0.]])

    def moments(amplitude):
        counts = paint_lightcone_particle_count_map_sparse(
            stencil, catalog, particle_mass_msun_h=1e10,
            concentration_params=ConcentrationParams(amplitude=amplitude), sample_chunk_size=2,
        )
        return angular_assignment_moments(
            vectors, counts / 1e4, vectors[0], lmax=20, pair_chunk_size=1
        )

    def prediction(amplitude):
        return moments(amplitude).self_pair

    derivative = jax.jacfwd(prediction)(5.71)
    finite_difference = (prediction(5.7101) - prediction(5.7099)) / 0.0002
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=2e-10)
    assert derivative[0] == pytest.approx(0., abs=1e-13)
    if ngp:
        np.testing.assert_array_equal(derivative, np.zeros(21))
    else:
        assert abs(derivative[-1]) > 1e-6

    model = ConstrainedBackboneAngularModel(
        jnp.ones(21), jnp.array([.2]), jnp.asarray(.3), jnp.array([.4]), jnp.ones(21),
    )

    def complete_prediction(amplitude):
        assignment = moments(amplitude)
        return jnp.stack(assemble_constrained_painted_angular_power(
            model, assignment.response[:, None], assignment.self_pair[:, None],
        ))

    complete_jvp = jax.jacfwd(complete_prediction)(5.71)
    complete_fd = (complete_prediction(5.7101) - complete_prediction(5.7099)) / .0002
    np.testing.assert_allclose(complete_jvp, complete_fd, rtol=2e-5, atol=2e-10)
    assert np.all(np.isfinite(complete_jvp))
    np.testing.assert_allclose(complete_jvp[:, 0], 0., atol=1e-13)
    if ngp:
        np.testing.assert_array_equal(complete_jvp, np.zeros((5, 21)))
    else:
        assert abs(complete_jvp[1, -1]) > 1e-8
        assert abs(complete_jvp[2, -1]) > 1e-8


def test_pixel_map_harmonics_match_exact_self_pair_without_a_pixel_window():
    hp = pytest.importorskip("healpy")
    nside, lmax = 16, 24
    pixels, weights = np.array([103, 145, 234]), np.array([0.2, 0.5, 0.3])
    vectors = np.array(hp.pix2vec(nside, pixels)).T
    moments = angular_assignment_moments(vectors, weights, vectors[0], lmax=lmax)
    values = np.zeros(hp.nside2npix(nside))
    values[pixels] = weights / hp.nside2pixarea(nside)
    measured = hp.anafast(values, lmax=lmax, iter=0)
    np.testing.assert_allclose(measured, moments.self_pair / (4 * np.pi), rtol=1e-12)


def _backbone():
    shot = jnp.array([0.1, 0.2])
    return HaloBackboneAngularSpectra(
        uncollapsed_auto=jnp.full(4, 2.),
        uncollapsed_halo_cross=jnp.broadcast_to(jnp.array([.3, -.1]), (4, 2)),
        halo_auto_cross=jnp.broadcast_to(jnp.array([[1., -.2], [-.2, .5]]), (4, 2, 2)),
        halo_self_pair=shot,
    )


def test_ngp_total_recovers_backbone_without_adding_noise_twice():
    backbone = _backbone()
    result = assemble_painted_angular_power(backbone, jnp.ones((4, 2)), jnp.ones((4, 2)))
    expected = (
        backbone.uncollapsed_auto + 2 * jnp.sum(backbone.uncollapsed_halo_cross, axis=1)
        + jnp.sum(backbone.halo_auto_cross, axis=(1, 2))
    )
    assert result.total.shape == (4,)
    np.testing.assert_allclose(result.total, expected, atol=1e-15)
    np.testing.assert_allclose(result.one_halo, .3)


def test_poisson_halo_population_uses_self_pair_not_squared_response():
    backbone = _backbone()._replace(
        uncollapsed_auto=jnp.zeros(4), uncollapsed_halo_cross=jnp.zeros((4, 2)),
        halo_auto_cross=jnp.broadcast_to(jnp.diag(jnp.array([.1, .2])), (4, 2, 2)),
    )
    result = assemble_painted_angular_power(backbone, jnp.full((4, 2), .2), jnp.full((4, 2), .5))
    np.testing.assert_allclose(result.distinct_halo, 0)
    np.testing.assert_allclose(result.total, .15)


def test_joint_covariance_recovers_linear_limit_despite_nonzero_halo_self_term():
    linear = jnp.array([10., 7., 3., 1.])
    mass_weighted_bias = jnp.array([.7, .2, .1])
    shot = jnp.array([.1, .2])
    coherent = linear[:, None, None] * (
        mass_weighted_bias[:, None] * mass_weighted_bias[None, :]
    )
    # Residual particle mass compensates the two independent halo residuals.
    residual_map = jnp.array([[-1., -1.], [1., 0.], [0., 1.]])
    stochastic = (residual_map * shot) @ residual_map.T
    covariance = coherent + stochastic[None, :, :]
    backbone = HaloBackboneAngularSpectra(
        covariance[:, 0, 0], covariance[:, 0, 1:], covariance[:, 1:, 1:], shot
    )
    result = assemble_painted_angular_power(backbone, jnp.ones((4, 2)), jnp.ones((4, 2)))
    np.testing.assert_allclose(result.total, linear, rtol=2e-15)
    np.testing.assert_allclose(result.one_halo, .3)


def test_assembly_gradient_and_empty_halo_population():
    derivative = jax.grad(lambda amplitude: jnp.sum(assemble_painted_angular_power(
        _backbone(), amplitude * jnp.ones((4, 2)), amplitude**2 * jnp.ones((4, 2))
    ).total))(1.)
    assert np.isfinite(derivative)
    empty = HaloBackboneAngularSpectra(
        jnp.ones(3), jnp.empty((3, 0)), jnp.empty((3, 0, 0)), jnp.empty(0)
    )
    np.testing.assert_allclose(assemble_painted_angular_power(
        empty, jnp.empty((3, 0)), jnp.empty((3, 0))
    ).total, 1)


def test_self_pair_amplitude_uses_total_shell_mean_and_squared_masses():
    value = catalogue_self_pair_amplitude(jnp.array([1., 3.]), 2., .01)
    assert value == pytest.approx(.01**2 * 10 / (4 * 4 * np.pi))
    assert catalogue_self_pair_amplitude(jnp.empty(0), 2., .01) == 0


@pytest.mark.parametrize("lmax,chunk", [(-1, 1), (1, 0)])
def test_invalid_static_controls(lmax, chunk):
    with pytest.raises(ValueError, match="lmax"):
        angular_assignment_moments(
            jnp.ones((1, 3)), jnp.ones(1), jnp.ones(3), lmax=lmax, pair_chunk_size=chunk
        )


def test_covariance_shape_validation():
    with pytest.raises(ValueError, match="dimensions"):
        assemble_painted_angular_power(_backbone(), jnp.ones((4, 3)), jnp.ones((4, 3)))


def test_production_branch_example_closes_all_three_assignments(tmp_path, monkeypatch):
    import importlib
    from pathlib import Path

    pytest.importorskip("healpy")
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("validate_painting_theory_operator")
    report = example.run(tmp_path, nside=8, lmax=12)
    for label, branch in [("NGP", "ngp"), ("Supersampled", "supersampled"), ("Native", "native")]:
        case = report["cases"][label]
        assert case["branch_counts"][branch] == 1
        assert case["global_mass_fraction"] == pytest.approx(1, abs=1e-12)
        assert case["quadrature_max_relative_error"] < 1e-10
        assert case["histogram_moment_max_absolute_error"] < 1e-10


def _constrained_backbone():
    return ConstrainedBackboneAngularModel(
        coherent_cl=jnp.array([10., 4., 1., .1]),
        halo_mass_weighted_bias=jnp.array([.2, .1]),
        uncollapsed_self_cl=jnp.asarray(.4),
        halo_self_cl=jnp.array([.5, .2]),
        constraint_window=jnp.array([1., .8, .2, 0.]),
    )


def test_constrained_contraction_matches_explicit_component_matrix():
    model = _constrained_backbone()
    noise = jnp.concatenate((model.uncollapsed_self_cl[None], model.halo_self_cl))
    bias = jnp.concatenate(((1 - model.halo_mass_weighted_bias.sum())[None],
                            model.halo_mass_weighted_bias))
    covariance = (model.coherent_cl[:, None, None] * (bias[:, None] * bias[None, :])
                  + jnp.diag(noise)[None, :, :]
                  - model.constraint_window[:, None, None]
                  * (noise[:, None] * noise[None, :]) / noise.sum())
    explicit = HaloBackboneAngularSpectra(
        covariance[:, 0, 0], covariance[:, 0, 1:], covariance[:, 1:, 1:], model.halo_self_cl,
    )
    response = jnp.array([[1., 1.], [.9, .95], [.6, .8], [.1, .2]])
    self_pair = response**2 + .01
    expected = assemble_painted_angular_power(explicit, response, self_pair)
    actual = jax.jit(assemble_constrained_painted_angular_power)(model, response, self_pair)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-13, atol=1e-14)
    assert actual.total.shape == (4,)
    assert np.all(np.asarray(actual.total) >= 0)
    assert np.all(np.linalg.eigvalsh(np.asarray(covariance)) >= -1e-14)


def test_constrained_linear_limit_and_small_scale_self_limit():
    model = _constrained_backbone()
    ones = jnp.ones((4, 2))
    result = assemble_constrained_painted_angular_power(model, ones, ones)
    total_self = model.uncollapsed_self_cl + model.halo_self_cl.sum()
    np.testing.assert_allclose(result.total, model.coherent_cl + total_self * (1-model.constraint_window))
    assert result.total[0] == pytest.approx(model.coherent_cl[0], abs=1e-13)
    small_scale = model._replace(coherent_cl=jnp.zeros(4), constraint_window=jnp.zeros(4))
    self_pair = jnp.full((4, 2), .5)
    high = assemble_constrained_painted_angular_power(small_scale, .3 * ones, self_pair)
    np.testing.assert_allclose(high.total, model.uncollapsed_self_cl + .5 * model.halo_self_cl.sum())
    np.testing.assert_allclose(high.particle_halo_cross, 0)
    np.testing.assert_allclose(high.distinct_halo, 0)


def test_constrained_cross_terms_have_profile_gradients():
    model = _constrained_backbone()

    def evaluate(amplitude):
        response = jnp.exp(-amplitude * jnp.arange(4)[:, None] * jnp.array([.1, .2]))
        return jnp.stack(assemble_constrained_painted_angular_power(model, response, response**2))

    actual = jax.jacfwd(evaluate)(.7)
    expected = (evaluate(.70001) - evaluate(.69999)) / .00002
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-10)
    assert np.all(np.isfinite(actual))
    assert np.any(np.asarray(actual[1, 1:]) != 0)
    assert np.any(np.asarray(actual[2, 1:]) != 0)
    np.testing.assert_array_equal(actual[:, 0], 0)


def test_constrained_zero_noise_and_empty_halo_bins():
    model = ConstrainedBackboneAngularModel(
        jnp.arange(3.), jnp.empty(0), jnp.asarray(0.), jnp.empty(0), jnp.ones(3),
    )
    result = assemble_constrained_painted_angular_power(model, jnp.empty((3, 0)), jnp.empty((3, 0)))
    np.testing.assert_array_equal(result.total, model.coherent_cl)


def test_constrained_shape_validation():
    with pytest.raises(ValueError, match="dimensions"):
        assemble_constrained_painted_angular_power(_constrained_backbone(), jnp.ones((4, 3)), jnp.ones((4, 3)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_constrained_total_preserves_small_linear_signal_under_large_cancellations(dtype):
    model = ConstrainedBackboneAngularModel(
        jnp.asarray([1e-20], dtype=dtype), jnp.asarray([.2], dtype=dtype),
        jnp.asarray(1e10, dtype=dtype), jnp.asarray([1e10], dtype=dtype),
        jnp.ones(1, dtype=dtype),
    )
    moments = jnp.ones((1, 1), dtype=dtype)
    total = assemble_constrained_painted_angular_power(model, moments, moments).total
    np.testing.assert_array_equal(total, model.coherent_cl)
