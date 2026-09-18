import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import geppetto.theory as theory_module
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology
from geppetto.io import (
    read_pinocchio_cosmology_table,
    read_pinocchio_mass_function_series,
)
from geppetto.profiles import NFWProfileParams, nfw_projected_surface_density
from geppetto.theory import (
    LINEAR_HIGH_ELL_FINITE_WIDTH,
    LINEAR_HIGH_ELL_LIMBER,
    HaloBiasTable,
    HaloMassFunctionTable,
    LinearPowerEvolutionTable,
    LinearTheoryTable,
    TwoHaloResponseTable,
    compensated_two_halo_response,
    exact_halo_model_shell_cls,
    exact_linear_shell_cls,
    finite_width_flat_sky_linear_shell_cls,
    finite_width_flat_sky_two_halo_shell_cls,
    gauss_legendre_rule,
    hybrid_angular_power_spectra,
    limber_halo_model_shell_cls,
    limber_shell_cls,
    linear_matter_power,
    linear_sigma_r,
    nfw_fourier_profile,
    one_halo_matter_power,
    particle_count_shot_noise,
    resolved_halo_mass_fraction,
    select_independent_limber_transitions,
    select_limber_transition,
    select_shell_high_ell_projection,
    sigma8_from_linear_power,
    spherical_top_hat_window,
    tabulate_shell_two_halo_response,
    two_halo_matter_power,
)


def _linear_theory() -> LinearTheoryTable:
    return LinearTheoryTable(
        h=0.7,
        omega_m0=0.3,
        scale_factor=jnp.asarray([0.5, 0.75, 1.0]),
        chi_mpc_h=jnp.asarray([1800.0, 800.0, 0.0]),
        omega_m=jnp.asarray([0.75, 0.5, 0.3]),
        growth=jnp.asarray([0.5, 0.75, 1.0]),
        k_h_mpc=jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]),
        power_mpc_h3=jnp.asarray([1.0e3, 5.0e3, 1.0e3, 50.0, 1.0]),
    )


def _mass_function() -> HaloMassFunctionTable:
    mass = jnp.asarray([1.0e12, 3.0e12, 1.0e13, 3.0e13, 1.0e14])
    shape = jnp.asarray([2.0e-3, 1.0e-3, 4.0e-4, 8.0e-5, 1.0e-5])
    return HaloMassFunctionTable(
        scale_factor=jnp.asarray([0.5, 1.0]),
        log_mass_msun_h=jnp.log(mass),
        dndlnm_mpc_h3=jnp.stack([0.7 * shape, shape]),
    )


def _separable_power_evolution() -> LinearPowerEvolutionTable:
    theory = _linear_theory()
    scale_factor = jnp.linspace(0.5, 1.0, 33)
    return LinearPowerEvolutionTable(
        scale_factor=scale_factor,
        k_h_mpc=theory.k_h_mpc,
        power_mpc_h3=scale_factor[:, None] ** 2 * theory.power_mpc_h3[None, :],
    )


def _halo_bias() -> HaloBiasTable:
    hmf = _mass_function()
    pbs = jnp.asarray(
        [
            [0.9, 1.0, 1.15, 1.35, 1.7],
            [0.8, 0.95, 1.1, 1.3, 1.6],
        ]
    )
    correction = jnp.full_like(pbs, 1.05)
    return HaloBiasTable(
        scale_factor=hmf.scale_factor,
        log_mass_msun_h=hmf.log_mass_msun_h,
        pbs_bias=pbs,
        correction=correction,
        linear_bias=pbs * correction,
    )


def test_linear_power_uses_pinocchio_growth_squared():
    theory = _linear_theory()
    present = linear_matter_power(jnp.asarray(0.1), 0.0, theory)
    redshift_one = linear_matter_power(jnp.asarray(0.1), 1.0, theory)

    assert present == pytest.approx(1000.0)
    assert redshift_one == pytest.approx(250.0)


def test_linear_power_uses_scale_dependent_pinocchio_table_when_supplied():
    theory = _linear_theory()
    evolution = LinearPowerEvolutionTable(
        scale_factor=jnp.asarray([0.5, 1.0]),
        k_h_mpc=jnp.asarray([0.01, 0.1]),
        power_mpc_h3=jnp.asarray([[100.0, 400.0], [1000.0, 1000.0]]),
    )

    power = linear_matter_power(
        jnp.asarray([0.01, 0.1]),
        jnp.asarray(1.0),
        theory,
        evolution,
    )
    gradient = jax.grad(
        lambda amplitude: jnp.sum(
            linear_matter_power(
                jnp.asarray([0.01, 0.1]),
                jnp.asarray(1.0),
                theory,
                evolution._replace(
                    power_mpc_h3=amplitude * evolution.power_mpc_h3
                ),
            )
        )
    )(1.0)

    np.testing.assert_allclose(power, [100.0, 400.0])
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_top_hat_sigma8_is_finite_differentiable_and_scales_with_power():
    theory = _linear_theory()
    sigma8 = sigma8_from_linear_power(theory)
    scaled = theory._replace(power_mpc_h3=4.0 * theory.power_mpc_h3)

    np.testing.assert_allclose(spherical_top_hat_window(jnp.asarray(0.0)), 1.0)
    np.testing.assert_allclose(sigma8_from_linear_power(scaled), 2.0 * sigma8, rtol=2.0e-6)
    assert jnp.isfinite(linear_sigma_r(jnp.asarray([4.0, 8.0]), theory)).all()
    gradient = jax.grad(
        lambda amplitude: sigma8_from_linear_power(
            theory._replace(power_mpc_h3=amplitude * theory.power_mpc_h3)
        )
    )(1.0)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_nfw_fourier_profile_is_normalized_and_differentiable():
    mass = jnp.asarray([1.0e13, 1.0e14])
    quadrature = gauss_legendre_rule(32)
    profile = nfw_fourier_profile(
        jnp.asarray([0.0, 0.5]),
        mass,
        jnp.asarray(0.3),
        Cosmology(),
        ConcentrationParams(),
        quadrature=quadrature,
    )

    assert profile.shape == (2, 2)
    np.testing.assert_allclose(profile[0], 1.0, rtol=0.0, atol=2.0e-7)
    gradient = jax.grad(
        lambda amplitude: jnp.sum(
            nfw_fourier_profile(
                jnp.asarray(0.5),
                mass,
                jnp.asarray(0.3),
                Cosmology(),
                ConcentrationParams(amplitude=amplitude),
                quadrature=quadrature,
            )
        )
    )(5.71)
    assert jnp.isfinite(gradient)


def test_one_halo_power_shape_gradient_and_ngp_independence():
    theory = _linear_theory()
    hmf = _mass_function()
    k = jnp.asarray([0.05, 0.5])
    power = one_halo_matter_power(
        k,
        0.2,
        theory,
        hmf,
        ConcentrationParams(),
        profile_quadrature=gauss_legendre_rule(16),
    )

    assert power.shape == (2,)
    assert jnp.all(jnp.isfinite(power))
    assert jnp.all(power > 0.0)
    gradient = jax.grad(
        lambda amplitude: jnp.sum(
            one_halo_matter_power(
                k,
                0.2,
                theory,
                hmf,
                ConcentrationParams(amplitude=amplitude),
                profile_quadrature=gauss_legendre_rule(16),
            )
        )
    )(5.71)
    assert jnp.isfinite(gradient)

    ngp_gradient = jax.grad(
        lambda amplitude: one_halo_matter_power(
            jnp.asarray(0.5),
            0.2,
            theory,
            hmf,
            ConcentrationParams(amplitude=amplitude),
            theta_resolution_rad=np.pi,
            profile_quadrature=gauss_legendre_rule(16),
        )
    )(5.71)
    assert ngp_gradient == pytest.approx(0.0, abs=1.0e-8)


def test_compensated_one_halo_power_vanishes_as_k_to_fourth():
    power_at_zero = one_halo_matter_power(
        jnp.asarray(0.0),
        0.2,
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        profile_quadrature=gauss_legendre_rule(32),
    )
    k = jnp.asarray([0.005, 0.01])
    low_k_power = one_halo_matter_power(
        k,
        0.2,
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        profile_quadrature=gauss_legendre_rule(32),
    )
    logarithmic_slope = jnp.log(low_k_power[1] / low_k_power[0]) / jnp.log(2.0)

    assert power_at_zero == pytest.approx(0.0, abs=1.0e-20)
    assert logarithmic_slope == pytest.approx(4.0, abs=0.08)


def test_compensated_two_halo_response_closes_at_low_k_and_has_finite_gradient():
    k = jnp.asarray([0.0, 0.2])

    def response_sum(amplitude):
        return jnp.sum(
            compensated_two_halo_response(
                k,
                0.2,
                _linear_theory(),
                _mass_function(),
                _halo_bias(),
                ConcentrationParams(amplitude=amplitude),
                profile_quadrature=gauss_legendre_rule(16),
            )
        )

    response = compensated_two_halo_response(
        k,
        0.2,
        _linear_theory(),
        _mass_function(),
        _halo_bias(),
        ConcentrationParams(),
        profile_quadrature=gauss_legendre_rule(16),
    )
    power = two_halo_matter_power(
        jnp.asarray([0.01, 0.2]),
        0.2,
        _linear_theory(),
        _mass_function(),
        _halo_bias(),
        ConcentrationParams(),
        profile_quadrature=gauss_legendre_rule(16),
    )

    assert response.shape == (2,)
    assert response[0] == pytest.approx(1.0, abs=1.0e-7)
    assert np.all(np.isfinite(power))
    assert jnp.isfinite(jax.grad(response_sum)(5.71))


def test_two_halo_response_table_and_limber_projection_shapes_and_gradient():
    ell = jnp.asarray([20, 40])

    def projected_sum(amplitude):
        response = tabulate_shell_two_halo_response(
            0.1,
            0.2,
            _linear_theory(),
            _mass_function(),
            _halo_bias(),
            ConcentrationParams(amplitude=amplitude),
            temporal_order=4,
            profile_quadrature=gauss_legendre_rule(8),
        )
        _, two_halo, _ = limber_halo_model_shell_cls(
            ell,
            0.1,
            0.2,
            _linear_theory(),
            _mass_function(),
            ConcentrationParams(amplitude=amplitude),
            response,
            radial_quadrature=gauss_legendre_rule(4),
            profile_quadrature=gauss_legendre_rule(8),
        )
        return jnp.sum(two_halo)

    response = tabulate_shell_two_halo_response(
        0.1,
        0.2,
        _linear_theory(),
        _mass_function(),
        _halo_bias(),
        ConcentrationParams(),
        temporal_order=4,
        profile_quadrature=gauss_legendre_rule(8),
    )

    assert response.response.shape == (4, 5)
    assert jnp.all(jnp.diff(response.scale_factor) > 0.0)
    assert jnp.isfinite(jax.grad(projected_sum)(5.71))


def test_resolved_mass_fraction_uses_only_measured_hmf_support():
    fraction = resolved_halo_mass_fraction(0.0, _mass_function(), Cosmology(omega_m=0.3))
    assert jnp.isfinite(fraction)
    assert 0.0 < fraction < 1.0


def test_limber_shell_cls_shape_and_concentration_gradient():
    ell = jnp.asarray([20, 40, 60])
    radial_rule = gauss_legendre_rule(8)
    profile_rule = gauss_legendre_rule(12)

    def one_halo_sum(amplitude):
        _, one_halo = limber_shell_cls(
            ell,
            0.1,
            0.2,
            _linear_theory(),
            _mass_function(),
            ConcentrationParams(amplitude=amplitude),
            radial_quadrature=radial_rule,
            profile_quadrature=profile_rule,
        )
        return jnp.sum(one_halo)

    linear, one_halo = limber_shell_cls(
        ell,
        0.1,
        0.2,
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        radial_quadrature=radial_rule,
        profile_quadrature=profile_rule,
    )
    assert linear.shape == one_halo.shape == (3,)
    assert jnp.all(jnp.isfinite(linear))
    assert jnp.all(jnp.isfinite(one_halo))
    assert jnp.isfinite(jax.grad(one_halo_sum)(5.71))


def test_scale_dependent_limber_reduces_to_scalar_growth_for_separable_power():
    arguments = (
        jnp.asarray([20, 40, 60]),
        0.1,
        0.2,
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
    )
    scalar, _ = limber_shell_cls(
        *arguments,
        radial_quadrature=gauss_legendre_rule(16),
        profile_quadrature=gauss_legendre_rule(8),
    )
    scale_dependent, _ = limber_shell_cls(
        *arguments,
        power_evolution=_separable_power_evolution(),
        radial_quadrature=gauss_legendre_rule(16),
        profile_quadrature=gauss_legendre_rule(8),
    )

    np.testing.assert_allclose(scale_dependent, scalar, rtol=5.0e-4)


def test_hybrid_spectra_shapes_weighted_one_halo_and_shot_noise():
    ell = jnp.asarray([20, 40])
    shell_weights = jnp.asarray([0.25, 0.75])
    result = hybrid_angular_power_spectra(
        ell,
        [0.1, 0.2],
        [0.2, 0.3],
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        [NFWProfileParams(), NFWProfileParams()],
        shell_weights=shell_weights,
        mean_uncollapsed_counts_per_pixel=jnp.asarray([5.0, 10.0]),
        mean_total_counts_per_pixel=jnp.asarray([10.0, 20.0]),
        pixel_area_sr=0.1,
        ell_exact_cap=0,
        radial_order=8,
        profile_order=12,
    )

    assert result.shell_total.shape == (2, 2)
    assert result.summed_total.shape == (2,)
    np.testing.assert_allclose(result.shell_two_halo, result.shell_linear)
    np.testing.assert_allclose(result.summed_two_halo, result.summed_linear)
    np.testing.assert_allclose(
        result.summed_one_halo,
        np.sum(np.asarray(shell_weights)[:, None] ** 2 * result.shell_one_halo, axis=0),
    )
    np.testing.assert_allclose(result.shell_particle_shot_noise[:, 0], [0.005, 0.0025])
    np.testing.assert_allclose(result.summed_particle_shot_noise, 0.1 * 15.0 / 30.0**2)
    assert result.ell_limber_start == 20


def test_hybrid_uses_corrected_two_halo_in_clustering_total():
    result = hybrid_angular_power_spectra(
        jnp.asarray([20, 40]),
        [0.1],
        [0.2],
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        [NFWProfileParams()],
        shell_weights=jnp.ones(1),
        halo_bias=_halo_bias(),
        ell_exact_cap=0,
        radial_order=4,
        finite_width_radial_order=8,
        finite_width_line_of_sight_order=8,
        profile_order=8,
    )

    assert result.shell_two_halo.shape == (1, 2)
    np.testing.assert_allclose(
        result.shell_clustering,
        result.shell_two_halo + result.shell_one_halo,
    )
    np.testing.assert_allclose(
        result.summed_clustering,
        result.summed_two_halo + result.summed_one_halo,
    )


def test_select_limber_transition_rejects_a_transient_match():
    ell = np.arange(20, 30)
    limber_shell = np.ones((2, ell.size))
    limber_sum = np.ones(ell.size)
    exact_shell = limber_shell.copy()
    exact_sum = limber_sum.copy()
    exact_shell[:, :3] *= 1.2
    exact_shell[1, 6] *= 1.02

    transition, shell_error, summed_error = select_limber_transition(
        ell,
        exact_shell,
        exact_sum,
        limber_shell,
        limber_sum,
        relative_tolerance=0.01,
        consecutive_multipoles=3,
    )

    assert transition == 27
    np.testing.assert_allclose(shell_error, 0.0)
    assert summed_error == pytest.approx(0.0)


def test_select_limber_transition_reports_no_match():
    ell = np.arange(2, 7)
    limber_shell = np.ones((1, ell.size))
    exact_shell = 1.1 * limber_shell

    transition, shell_error, summed_error = select_limber_transition(
        ell,
        exact_shell,
        np.full(ell.size, 1.1),
        limber_shell,
        np.ones(ell.size),
        relative_tolerance=0.01,
        consecutive_multipoles=2,
    )

    assert transition is None
    np.testing.assert_allclose(shell_error, [1.0 - 1.0 / 1.1])
    assert summed_error == pytest.approx(1.0 - 1.0 / 1.1)


def test_independent_limber_transitions_allow_different_shell_switches():
    ell = np.arange(2, 11)
    limber_shell = np.ones((2, ell.size))
    exact_shell = limber_shell.copy()
    exact_shell[0, :2] = 1.2
    exact_shell[1, :4] = 1.3
    limber_sum = np.ones(ell.size)
    exact_sum = limber_sum.copy()
    exact_sum[:3] = 1.4

    shell_transition, summed_transition, shell_error, summed_error = (
        select_independent_limber_transitions(
            ell,
            exact_shell,
            exact_sum,
            limber_shell,
            limber_sum,
            relative_tolerance=0.01,
            consecutive_multipoles=2,
        )
    )

    np.testing.assert_array_equal(shell_transition, [4, 6])
    assert summed_transition == 5
    np.testing.assert_allclose(shell_error, 0.0)
    assert summed_error == pytest.approx(0.0)


def test_shell_high_ell_projection_selects_smallest_final_error():
    exact = np.ones((2, 5))
    limber = exact.copy()
    finite_width = exact.copy()
    limber[0, -2:] *= 1.08
    finite_width[1, -2:] *= 1.05

    selected, modes = select_shell_high_ell_projection(
        exact,
        np.stack((limber, finite_width)),
        np.asarray((LINEAR_HIGH_ELL_LIMBER, LINEAR_HIGH_ELL_FINITE_WIDTH)),
        comparison_width=2,
    )

    np.testing.assert_allclose(selected, exact)
    np.testing.assert_array_equal(
        modes,
        [LINEAR_HIGH_ELL_FINITE_WIDTH, LINEAR_HIGH_ELL_LIMBER],
    )


def test_hybrid_projection_applies_independent_transitions(monkeypatch):
    def fake_limber(ell, *args, **kwargs):
        values = jnp.ones_like(jnp.asarray(ell), dtype=jnp.float32)
        return values, jnp.zeros_like(values)

    def fake_exact(ell, z_lo, *args, **kwargs):
        ell_values = np.asarray(ell)
        shell = np.stack(
            (
                np.where(ell_values < 4, 2.0, 1.0),
                np.where(ell_values < 6, 3.0, 1.0),
            )
        )
        summed = np.where(ell_values < 5, 2.0, 0.5)
        return shell, summed

    monkeypatch.setattr(theory_module, "limber_shell_cls", fake_limber)
    monkeypatch.setattr(
        theory_module,
        "finite_width_flat_sky_linear_shell_cls",
        lambda ell, *args, **kwargs: jnp.ones_like(jnp.asarray(ell), dtype=jnp.float32),
    )
    monkeypatch.setattr(theory_module, "exact_linear_shell_cls", fake_exact)
    result = hybrid_angular_power_spectra(
        jnp.arange(2, 11),
        [0.1, 0.2],
        [0.2, 0.3],
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        [NFWProfileParams(), NFWProfileParams()],
        shell_weights=jnp.ones(2),
        ell_exact_cap=8,
        limber_match_width=2,
        exact_batch_size=7,
    )

    np.testing.assert_array_equal(result.shell_ell_high_ell_start, [4, 6])
    assert result.summed_ell_limber_start == 5
    np.testing.assert_allclose(
        result.shell_linear[0], [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        result.shell_linear[1], [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        result.summed_linear,
        [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )


def test_hybrid_projection_fails_when_limber_does_not_match_before_cap(monkeypatch):
    def fake_limber(ell, *args, **kwargs):
        values = jnp.ones_like(jnp.asarray(ell), dtype=jnp.float32)
        return values, jnp.zeros_like(values)

    def fake_exact(ell, z_lo, *args, **kwargs):
        values = np.full((len(z_lo), len(ell)), 1.2)
        return values, np.full(len(ell), 1.2)

    monkeypatch.setattr(theory_module, "limber_shell_cls", fake_limber)
    monkeypatch.setattr(
        theory_module,
        "finite_width_flat_sky_linear_shell_cls",
        lambda ell, *args, **kwargs: jnp.ones_like(jnp.asarray(ell), dtype=jnp.float32),
    )
    monkeypatch.setattr(theory_module, "exact_linear_shell_cls", fake_exact)

    with pytest.raises(ValueError, match=r"final_window=5-6"):
        hybrid_angular_power_spectra(
            jnp.arange(2, 11),
            [0.1],
            [0.2],
            _linear_theory(),
            _mass_function(),
            ConcentrationParams(),
            [NFWProfileParams()],
            shell_weights=jnp.ones(1),
            ell_exact_cap=6,
            limber_match_width=2,
            exact_batch_size=2,
        )


def test_particle_count_shot_noise_formula():
    noise = particle_count_shot_noise(jnp.asarray(4.0), jnp.asarray(10.0), 0.2)
    assert noise == pytest.approx(0.008)


def test_exact_linear_projection_converges_to_limber_at_switch():
    pytest.importorskip("scipy")
    case = Path(__file__).parents[1] / "examples" / "pinocchio_geppetto_case"
    theory = read_pinocchio_cosmology_table(case / "pinocchio.example.cosmology.out")
    hmf = read_pinocchio_mass_function_series(tuple(sorted(case.glob("*.mf.out"))))
    exact, exact_sum = exact_linear_shell_cls(
        np.asarray([100]),
        np.asarray([0.05]),
        np.asarray([0.1]),
        theory,
        radial_order=64,
        relative_tolerance=5.0e-4,
    )
    limber, _ = limber_shell_cls(
        jnp.asarray([100]),
        0.05,
        0.1,
        theory,
        hmf,
        ConcentrationParams(),
        radial_quadrature=gauss_legendre_rule(32),
        profile_quadrature=gauss_legendre_rule(8),
    )

    assert exact.shape == (1, 1)
    np.testing.assert_allclose(exact_sum, exact[0], rtol=1.0e-8)
    np.testing.assert_allclose(exact[0], limber, rtol=0.1)


def test_finite_width_projection_matches_exact_for_thin_shell():
    pytest.importorskip("scipy")
    case = Path(__file__).parents[1] / "examples" / "pinocchio_geppetto_case"
    theory = read_pinocchio_cosmology_table(case / "pinocchio.example.cosmology.out")
    ell = np.asarray([100])
    exact, _ = exact_linear_shell_cls(
        ell,
        np.asarray([1.9]),
        np.asarray([2.0]),
        theory,
        radial_order=256,
        radial_tail_periods=80,
    )
    finite_width = finite_width_flat_sky_linear_shell_cls(
        jnp.asarray(ell),
        1.9,
        2.0,
        theory,
        radial_quadrature=gauss_legendre_rule(128),
        line_of_sight_quadrature=gauss_legendre_rule(256),
        line_of_sight_tail_periods=40,
    )

    def projected_sum(amplitude):
        scaled = theory._replace(
            power_mpc_h3=amplitude * theory.power_mpc_h3,
        )
        return jnp.sum(
            finite_width_flat_sky_linear_shell_cls(
                jnp.asarray(ell),
                1.9,
                2.0,
                scaled,
                radial_quadrature=gauss_legendre_rule(32),
                line_of_sight_quadrature=gauss_legendre_rule(64),
                line_of_sight_tail_periods=20,
            )
        )

    assert finite_width.shape == (1,)
    assert jnp.isfinite(jax.grad(projected_sum)(jnp.asarray(1.0)))
    np.testing.assert_allclose(finite_width, exact[0], rtol=0.01)


def test_scale_dependent_finite_width_reduces_to_separable_growth():
    ell = jnp.asarray([40, 80])
    common = {
        "radial_quadrature": gauss_legendre_rule(32),
        "line_of_sight_quadrature": gauss_legendre_rule(64),
        "line_of_sight_tail_periods": 20,
    }
    scalar = finite_width_flat_sky_linear_shell_cls(
        ell,
        0.1,
        0.2,
        _linear_theory(),
        **common,
    )
    scale_dependent = finite_width_flat_sky_linear_shell_cls(
        ell,
        0.1,
        0.2,
        _linear_theory(),
        power_evolution=_separable_power_evolution(),
        **common,
    )

    np.testing.assert_allclose(scale_dependent, scalar, rtol=8.0e-4)


def test_finite_width_two_halo_applies_response_inside_radial_transfer():
    theory = _linear_theory()
    response = TwoHaloResponseTable(
        scale_factor=jnp.asarray([0.5, 1.0]),
        k_h_mpc=theory.k_h_mpc,
        response=jnp.full((2, theory.k_h_mpc.size), 2.0),
    )
    common = {
        "radial_quadrature": gauss_legendre_rule(32),
        "line_of_sight_quadrature": gauss_legendre_rule(64),
        "line_of_sight_tail_periods": 20,
    }
    linear = finite_width_flat_sky_linear_shell_cls(
        jnp.asarray([40, 80]),
        0.1,
        0.2,
        theory,
        **common,
    )
    two_halo = finite_width_flat_sky_two_halo_shell_cls(
        jnp.asarray([40, 80]),
        0.1,
        0.2,
        theory,
        response,
        **common,
    )

    np.testing.assert_allclose(two_halo, 4.0 * linear, rtol=8.0e-4)


def test_scale_dependent_exact_projection_reduces_to_separable_growth():
    pytest.importorskip("scipy")
    arguments = (
        np.asarray([20]),
        np.asarray([0.1]),
        np.asarray([0.2]),
        _linear_theory(),
    )
    scalar = exact_linear_shell_cls(
        *arguments,
        radial_order=32,
        radial_tail_periods=40,
        relative_tolerance=1.0e-3,
    )
    scale_dependent = exact_linear_shell_cls(
        *arguments,
        power_evolution=_separable_power_evolution(),
        radial_order=32,
        radial_tail_periods=40,
        relative_tolerance=1.0e-3,
    )

    np.testing.assert_allclose(scale_dependent[0], scalar[0], rtol=1.0e-3)
    np.testing.assert_allclose(scale_dependent[1], scalar[1], rtol=1.0e-3)


def test_exact_two_halo_applies_constant_response_to_shell_and_cross_spectra():
    pytest.importorskip("scipy")
    theory = _linear_theory()
    response = TwoHaloResponseTable(
        scale_factor=jnp.asarray([0.5, 1.0]),
        k_h_mpc=theory.k_h_mpc,
        response=jnp.full((2, theory.k_h_mpc.size), 2.0),
    )
    linear_shell, linear_sum, two_halo_shell, two_halo_sum = exact_halo_model_shell_cls(
        np.asarray([20]),
        np.asarray([0.1]),
        np.asarray([0.2]),
        theory,
        (response,),
        radial_order=32,
        radial_tail_periods=40,
        relative_tolerance=1.0e-3,
    )

    np.testing.assert_allclose(two_halo_shell, 4.0 * linear_shell, rtol=1.0e-3)
    np.testing.assert_allclose(two_halo_sum, 4.0 * linear_sum, rtol=1.0e-3)


def test_exact_near_observer_shell_requires_converged_radial_order():
    pytest.importorskip("scipy")
    case = Path(__file__).parents[1] / "examples" / "pinocchio_geppetto_case"
    theory = read_pinocchio_cosmology_table(case / "pinocchio.example.cosmology.out")
    arguments = (
        np.asarray([256]),
        np.asarray([0.0]),
        np.asarray([0.05]),
        theory,
    )

    under_resolved, _ = exact_linear_shell_cls(
        *arguments,
        radial_order=64,
        radial_tail_periods=128,
    )
    converged, _ = exact_linear_shell_cls(
        *arguments,
        radial_order=256,
        radial_tail_periods=128,
    )
    reference, _ = exact_linear_shell_cls(
        *arguments,
        radial_order=384,
        radial_tail_periods=128,
    )

    np.testing.assert_allclose(converged, reference, rtol=1.0e-8, atol=0.0)
    assert abs(under_resolved[0, 0] / reference[0, 0] - 1.0) > 0.1


def test_exact_projection_rejects_too_short_radial_tail():
    with pytest.raises(ValueError, match="at least 40"):
        exact_linear_shell_cls(
            np.asarray([2]),
            np.asarray([0.1]),
            np.asarray([0.2]),
            _linear_theory(),
            radial_tail_periods=39.0,
        )


def test_exact_projection_rejects_invalid_power_evolution_axes():
    evolution = _separable_power_evolution()._replace(
        k_h_mpc=jnp.asarray([0.0, 0.01, 0.1, 1.0, 10.0])
    )

    with pytest.raises(ValueError, match="positive finite P"):
        exact_linear_shell_cls(
            np.asarray([2]),
            np.asarray([0.1]),
            np.asarray([0.2]),
            _linear_theory(),
            power_evolution=evolution,
            radial_order=8,
            radial_tail_periods=40.0,
        )


def test_exact_linear_projection_process_workers_preserve_results(monkeypatch):
    pytest.importorskip("scipy")
    case = Path(__file__).parents[1] / "examples" / "pinocchio_geppetto_case"
    theory = read_pinocchio_cosmology_table(case / "pinocchio.example.cosmology.out")
    arguments = (
        np.asarray([2, 3]),
        np.asarray([0.05]),
        np.asarray([0.1]),
        theory,
    )

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_PLACES", "cores")
    monkeypatch.setenv("OMP_PROC_BIND", "spread")
    serial = exact_linear_shell_cls(
        *arguments,
        radial_order=16,
        radial_tail_periods=40,
        workers=1,
    )
    parallel = exact_linear_shell_cls(
        *arguments,
        radial_order=16,
        radial_tail_periods=40,
        workers=2,
    )

    np.testing.assert_allclose(parallel[0], serial[0], rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(parallel[1], serial[1], rtol=1.0e-12, atol=0.0)
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["OMP_PLACES"] == "cores"
    assert os.environ["OMP_PROC_BIND"] == "spread"


def test_nfw_3d_transform_matches_projected_profile_hankel_transform():
    scipy = pytest.importorskip("scipy.special")
    mass = jnp.asarray(1.0e14)
    redshift = jnp.asarray(0.3)
    concentration = ConcentrationParams()
    profile_params = NFWProfileParams()
    cosmology = Cosmology()
    from geppetto.profiles import nfw_scale_radius_and_density

    r_delta, _, _, _ = nfw_scale_radius_and_density(
        mass,
        redshift,
        cosmology,
        concentration,
        profile_params,
    )
    radius = jnp.linspace(0.0, r_delta, 20_001)
    sigma = nfw_projected_surface_density(
        radius,
        mass,
        redshift,
        cosmology,
        concentration,
        profile_params,
    )
    k = 0.5
    radial_weight = np.asarray(radius * sigma)
    normalization = np.trapezoid(radial_weight, np.asarray(radius))
    hankel = (
        np.trapezoid(
            radial_weight * scipy.j0(k * np.asarray(radius)),
            np.asarray(radius),
        )
        / normalization
    )
    transform = nfw_fourier_profile(
        jnp.asarray(k),
        jnp.asarray([mass]),
        redshift,
        cosmology,
        concentration,
        profile_params,
        quadrature=gauss_legendre_rule(64),
    )
    np.testing.assert_allclose(transform[0], hankel, rtol=2.0e-3)
